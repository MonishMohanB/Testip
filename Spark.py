import logging
import subprocess
import shlex
import sys
from typing import Dict, List, Optional, Union

class SparkSubmit:
    def __init__(
        self,
        spark_submit_path: str = "spark-submit",
        master: Optional[str] = None,
        deploy_mode: Optional[str] = None,
        application_path: Optional[str] = None,
        app_arguments: Optional[List[str]] = None,
        app_class: Optional[str] = None,
        app_name: Optional[str] = None,
        conf: Optional[Dict[str, str]] = None,
        jars: Optional[List[str]] = None,
        py_files: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        archives: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.spark_submit_path = spark_submit_path
        self.master = master
        self.deploy_mode = deploy_mode
        self.application_path = application_path
        self.app_arguments = app_arguments if app_arguments is not None else []
        self.app_class = app_class
        self.app_name = app_name
        self.conf = conf if conf is not None else {}
        self.jars = jars if jars is not None else []
        self.py_files = py_files if py_files is not None else []
        self.files = files if files is not None else []
        self.archives = archives if archives is not None else []

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("SparkSubmit")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _build_command(self) -> List[str]:
        command = [self.spark_submit_path]

        if self.master:
            command.extend(["--master", self.master])
        if self.deploy_mode:
            command.extend(["--deploy-mode", self.deploy_mode])
        if self.app_class:
            command.extend(["--class", self.app_class])
        if self.app_name:
            command.extend(["--name", self.app_name])

        for key, value in self.conf.items():
            command.extend(["--conf", f"{key}={value}"])

        if self.jars:
            command.extend(["--jars", ",".join(self.jars)])
        if self.py_files:
            command.extend(["--py-files", ",".join(self.py_files)])
        if self.files:
            command.extend(["--files", ",".join(self.files)])
        if self.archives:
            command.extend(["--archives", ",".join(self.archives)])

        if self.application_path:
            command.append(self.application_path)

        command.extend(self.app_arguments)

        return command

    def submit_job(self) -> int:
        try:
            command = self._build_command()
            self.logger.info(f"Constructed command: {' '.join(shlex.quote(arg) for arg in command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self.logger.info("Spark job submitted. Capturing output...")

            # Capture and log stdout and stderr in real-time
            while True:
                if process.stdout is not None:
                    for line in process.stdout:
                        self.logger.info(f"STDOUT: {line.strip()}")
                if process.stderr is not None:
                    for line in process.stderr:
                        self.logger.error(f"STDERR: {line.strip()}")

                # Check if process has terminated
                if process.poll() is not None:
                    break

            return_code = process.wait()
            self.logger.info(f"Job completed with return code: {return_code}")
            return return_code

        except FileNotFoundError:
            self.logger.exception(f"Spark-submit executable not found at '{self.spark_submit_path}'. Ensure Spark is installed and the path is correct.")
            raise
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger("SparkSubmitExample")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Configuration
    spark_args = {
        "spark_submit_path": "/path/to/spark-submit",
        "master": "yarn",
        "deploy_mode": "cluster",
        "application_path": "/path/to/your/app.py",
        "app_arguments": ["arg1", "arg2"],
        "app_class": "com.example.YourAppClass",
        "app_name": "Example Spark Job",
        "conf": {
            "spark.executor.memory": "4g",
            "spark.driver.memory": "2g"
        },
        "jars": ["/path/to/dependency1.jar", "/path/to/dependency2.jar"],
        "py_files": ["/path/to/helper.py"],
        "logger": logger,
    }

    # Submit job
    spark_job = SparkSubmit(**spark_args)
    return_code = spark_job.submit_job()
    logger.info(f"Spark job exited with return code: {return_code}")




import logging
import subprocess
import shlex
import sys
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class SparkSubmit:
    def __init__(
        self,
        spark_submit_path: str = "spark-submit",
        master: Optional[str] = None,
        deploy_mode: Optional[str] = None,
        application_path: Optional[str] = None,
        app_arguments: Optional[List[str]] = None,
        app_class: Optional[str] = None,
        app_name: Optional[str] = None,
        conf: Optional[Dict[str, str]] = None,
        jars: Optional[List[str]] = None,
        py_files: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        archives: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.spark_submit_path = spark_submit_path
        self.master = master
        self.deploy_mode = deploy_mode
        self.application_path = application_path
        self.app_arguments = app_arguments if app_arguments is not None else []
        self.app_class = app_class
        self.app_name = app_name
        self.conf = conf if conf is not None else {}
        self.jars = jars if jars is not None else []
        self.py_files = py_files if py_files is not None else []
        self.files = files if files is not None else []
        self.archives = archives if archives is not None else []

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("SparkSubmit")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _build_command(self) -> List[str]:
        command = [self.spark_submit_path]

        if self.master:
            command.extend(["--master", self.master])
        if self.deploy_mode:
            command.extend(["--deploy-mode", self.deploy_mode])
        if self.app_class:
            command.extend(["--class", self.app_class])
        if self.app_name:
            command.extend(["--name", self.app_name])

        for key, value in self.conf.items():
            command.extend(["--conf", f"{key}={value}"])

        if self.jars:
            command.extend(["--jars", ",".join(self.jars)])
        if self.py_files:
            command.extend(["--py-files", ",".join(self.py_files)])
        if self.files:
            command.extend(["--files", ",".join(self.files)])
        if self.archives:
            command.extend(["--archives", ",".join(self.archives)])

        if self.application_path:
            command.append(self.application_path)

        command.extend(self.app_arguments)

        return command

    def submit_job(self) -> int:
        try:
            command = self._build_command()
            self.logger.info(f"Constructed command: {' '.join(shlex.quote(arg) for arg in command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self.logger.info("Spark job submitted. Capturing output...")

            # Capture and log stdout and stderr in real-time
            while True:
                if process.stdout is not None:
                    for line in process.stdout:
                        self.logger.info(f"STDOUT: {line.strip()}")
                if process.stderr is not None:
                    for line in process.stderr:
                        self.logger.error(f"STDERR: {line.strip()}")

                # Check if process has terminated
                if process.poll() is not None:
                    break

            return_code = process.wait()
            self.logger.info(f"Job completed with return code: {return_code}")
            return return_code

        except FileNotFoundError:
            self.logger.exception(f"Spark-submit executable not found at '{self.spark_submit_path}'. Ensure Spark is installed and the path is correct.")
            raise
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred: {str(e)}")
            raise

class ConcurrentSparkSubmit:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.results = {}

    def submit_jobs(self, job_configs: List[Dict]) -> Dict[str, Union[int, Exception]]:
        """Submit multiple Spark jobs concurrently.
        
        Args:
            job_configs: List of dictionaries, each containing configuration for a Spark job.
            
        Returns:
            Dictionary with job names as keys and return codes or exceptions as values.
        """
        self.results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {}
            for config in job_configs:
                # Create a dedicated logger for each job
                job_name = config.get('app_name', f"job_{time.time()}")
                logger = self._create_job_logger(job_name)
                config['logger'] = logger
                
                # Submit job to thread pool
                future = executor.submit(self._run_single_job, config)
                future_to_job[future] = job_name

            # Process results as jobs complete
            for future in as_completed(future_to_job):
                job_name = future_to_job[future]
                try:
                    return_code = future.result()
                    self.results[job_name] = return_code
                except Exception as e:
                    self.results[job_name] = e
        return self.results

    def _create_job_logger(self, job_name: str) -> logging.Logger:
        """Create a dedicated logger for a job."""
        logger = logging.getLogger(f"SparkSubmit.{job_name}")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent propagation to root logger
        return logger

    def _run_single_job(self, config: Dict) -> int:
        """Wrapper to run a single Spark job and handle exceptions."""
        job = SparkSubmit(**config)
        try:
            return job.submit_job()
        except Exception as e:
            config['logger'].exception(f"Job failed with exception: {str(e)}")
            raise

if __name__ == "__main__":
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Generate configurations for 10 different jobs
    job_configs = []
    for i in range(1, 11):
        config = {
            "spark_submit_path": "/path/to/spark-submit",
            "master": "yarn",
            "deploy_mode": "cluster",
            "application_path": f"/path/to/app_{i}.py",
            "app_arguments": [f"arg{i}_1", f"arg{i}_2"],
            "app_name": f"Job_{i}",
            "conf": {
                "spark.executor.memory": f"{i}g",
                "spark.driver.memory": f"{max(1, i//2)}g"
            },
            "jars": [f"/path/to/job_{i}_lib.jar"],
        }
        job_configs.append(config)

    # Submit jobs concurrently
    concurrent_runner = ConcurrentSparkSubmit(max_workers=5)
    results = concurrent_runner.submit_jobs(job_configs)

    # Print job results
    root_logger.info("\nJob Execution Summary:")
    for job_name, result in results.items():
        if isinstance(result, Exception):
            root_logger.error(f"{job_name}: FAILED ({type(result).__name__})")
        else:
            root_logger.info(f"{job_name}: RETURN CODE {result}")
