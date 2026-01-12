"""
Background workers for camera calibration optimisation.
These run in separate processes and communicate via queues using serialized dicts.
"""
import multiprocessing
import queue
import time
from typing import Dict, Any

from core.genetic_algorithm import run_genetic_step
from core.refinement import run_refinement


class GAWorker(multiprocessing.Process):
    """Runs genetic algorithm for camera calibration in separate process."""

    def __init__(
        self,
        command_queue: multiprocessing.Queue,
        progress_queue: multiprocessing.Queue
    ):
        super().__init__(name="GAWorker", daemon=True)
        self.command_queue = command_queue
        self.progress_queue = progress_queue

    def run(self):
        print("GA worker started.")
        ga_state: Dict[str, Any] = {}
        is_running = False

        while True:
            # Check for commands (non-blocking)
            try:
                command = self.command_queue.get_nowait()
                action = command.get("action")

                if action == "shutdown":
                    break

                elif action == "start":
                    print("GA worker received start command.")
                    ga_state = command.get("snapshot", {})
                    ga_state["population"] = None  # Reset population for fresh start
                    is_running = True

                elif action == "stop":
                    print("GA worker received stop command.")
                    is_running = False

            except queue.Empty:
                pass

            # Run GA step if active
            if is_running:
                result = run_genetic_step(ga_state)

                # Update state for next iteration
                ga_state["best_fitness"] = result["new_best_fitness"]
                ga_state["best_individual"] = result["new_best_individual"]
                ga_state["generation"] = result["generation"]
                ga_state["population"] = result["next_population"]
                ga_state["stagnation_counter"] = result["stagnation_counter"]

                # Send progress (dict only - no CameraRig objects cross process boundary)
                self.progress_queue.put({
                    "status": "running",
                    "generation": result["generation"],
                    "best_fitness": result["new_best_fitness"],
                    "mean_fitness": result["mean_fitness"],
                    "new_best_individual": result["new_best_individual"]  # Serialized dict
                })
            else:
                time.sleep(0.01)

        print("GA worker shut down.")


class BAWorker(multiprocessing.Process):
    """Runs bundle adjustment in a separate process."""

    def __init__(
        self,
        command_queue: multiprocessing.Queue,
        results_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event
    ):
        super().__init__(name="BAWorker", daemon=True)
        self.command_queue = command_queue
        self.results_queue = results_queue
        self.stop_event = stop_event

    def run(self):
        print("BA worker started.")

        while True:
            try:
                command = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            action = command.get("action")

            if action == "shutdown":
                break

            if action == "start":
                print("BA worker received start command.")
                snapshot = command.get("snapshot", {})
                self.stop_event.clear()

                try:
                    result = run_refinement(snapshot)

                    if self.stop_event.is_set():
                        print("BA completed but stop was requested - discarding results.")
                    else:
                        self.results_queue.put(result)

                except Exception as e:
                    import traceback
                    print("[ BA WORKER EXCEPTION ]")
                    traceback.print_exc()
                    print("[   END EXCEPTION     ]")

                    if not self.stop_event.is_set():
                        self.results_queue.put({"status": "error", "message": str(e)})

        print("BA worker shut down.")