import multiprocessing
import queue
import time
from core import run_genetic_step, run_refinement


class GAWorker(multiprocessing.Process):
    """Runs genetic algorithm for camera calibration in separate process."""

    def __init__(self, command_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue):
        super().__init__(name="GAWorker")
        self.command_queue = command_queue
        self.progress_queue = progress_queue
        self.ga_state = {"is_running": False}

    def run(self):
        print("GA worker started.")

        while True:
            # Check for commands
            try:
                command = self.command_queue.get_nowait()

                if command.get("action") == "shutdown":
                    break
                elif command.get("action") == "start":
                    print("GA worker received start command.")
                    self.ga_state = command.get("ga_state_snapshot")
                    self.ga_state["is_running"] = True
                    self.ga_state["population"] = None  # Reset population
                elif command.get("action") == "stop":
                    print("GA worker received stop command.")
                    self.ga_state["is_running"] = False

            except queue.Empty:
                pass

            # Run GA step if active
            if self.ga_state.get("is_running"):
                progress = run_genetic_step(self.ga_state)

                # Update state for next iteration
                self.ga_state["best_fitness"] = progress["new_best_fitness"]
                self.ga_state["best_individual"] = progress["new_best_individual"]
                self.ga_state["generation"] = progress["generation"]
                self.ga_state["population"] = progress["next_population"]

                progress_payload = {
                    "status": "running",
                    "generation": progress["generation"],
                    "best_fitness": progress["new_best_fitness"],
                    "mean_fitness": progress["mean_fitness"],
                    "new_best_individual": progress["new_best_individual"]
                }
                self.progress_queue.put(progress_payload)
            else:
                time.sleep(0.01)

        print("GA worker shut down.")


class BAWorker(multiprocessing.Process):
    """Runs bundle adjustment in a separate process."""

    def __init__(self, command_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
        super().__init__(name="BAWorker")
        self.command_queue = command_queue
        self.results_queue = results_queue
        self.stop_event = stop_event

    def run(self):
        print("BA worker started.")
        while True:

            try:
                command = self.command_queue.get()  # Block until a command arrives

                if command.get("action") == "shutdown":
                    break

                if command.get("action") == "start":
                    print("BA worker received start command.")
                    snapshot = command.get("ba_state_snapshot")

                    try:
                        results = run_refinement(snapshot)

                        if self.stop_event.is_set():
                            print("BA worker finished, but a stop was requested. Discarding results.")
                        else:
                            # if not cancelled send the results
                            self.results_queue.put(results)

                    except Exception as e:
                        import traceback

                        print(f"[ BA WORKER EXCEPTION ]")
                        traceback.print_exc()
                        print(f"[     END EXCEPTION   ]")

                        # check for cancellation before reporting an error
                        if not self.stop_event.is_set():
                            self.results_queue.put({"status": "error", "message": str(e)})

            except queue.Empty:
                continue

        print("BA worker shut down.")
