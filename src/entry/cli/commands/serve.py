import os
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn

from config import GlobalConfig
from services.index import MilvusDatabase

from .command import BaseCommand


class ServeCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super(ServeCommand, self).__init__(*args, **kwargs)

    def add_args(self, subparser):
        parser = subparser.add_parser("serve", help="Start RestAPI and WebUI")
        parser.add_argument(
            "-p",
            "--port",
            dest="port",
            default=5100,
            type=int,
        )
        parser.add_argument(
            "-d",
            "--dev",
            dest="dev_mode",
            action="store_true",
            help="Use dev mode",
        )
        parser.add_argument(
            "-w",
            "--workers",
            dest="workers",
            type=int,
            default=1,
            help="Number of workers to serve in uvicorn",
        )
        parser.add_argument(
            "--reload",
            dest="reload",
            action="store_true",
            help="Enable auto-reload on code changes (for development)",
        )
        parser.add_argument(
            "--database",
            dest="database_type",
            type=str,
            choices=["faiss", "milvus"],
            help="Database type to use (faiss or milvus). Defaults to config setting",
        )

        parser.set_defaults(func=self)

    def __call__(
        self, port, dev_mode, workers, reload=False, database_type=None, *args, **kwargs
    ):
        db_type = database_type or GlobalConfig.get("webui", "database") or "faiss"

        if db_type.lower() == "milvus":
            MilvusDatabase.start_server()

        if database_type:
            GlobalConfig.set(database_type, "webui", "database")

        self._install_frontend()
        if len(GlobalConfig.get("webui", "features") or []) == 0:
            self._logger.error(
                f'No models found in "{GlobalConfig.RELATED_CONFIG_FILE_PATH}". Check your "{GlobalConfig.RELATED_CONFIG_FILE_PATH}"'
            )
            sys.exit(1)

        self._logger.info(f"Starting web server with {db_type} database backend")
        os.environ["AIC25_WORK_DIR"] = str(self._work_dir)

        # Configure reload parameters for development
        reload_params = {}
        if reload or dev_mode:
            src_dir = Path(__file__).parent.parent.parent  # Points to src/
            reload_dirs = [
                str(src_dir / "entry"),  # FastAPI app code
                str(src_dir / "services"),  # Services code
                str(src_dir / "config"),  # Configuration code
                str(src_dir / "utils"),  # Utilities
            ]
            reload_params = {
                "reload": True,
                "reload_dirs": reload_dirs,
            }
            if reload:
                self._logger.info(
                    "Auto-reload enabled. Server will restart on code changes."
                )

        params = {}
        if dev_mode:
            params = {**params, **reload_params}
            dev_cmd = ["npm", "run", "dev"]
            dev_env = os.environ.copy()
            dev_env["VITE_PORT"] = str(port)

            p = subprocess.Popen(
                dev_cmd,
                env=dev_env,
                cwd=str(Path(__file__).parent / "../../../entry/web/view"),
            )
        else:
            if reload:
                params = {**params, **reload_params}
            self._build_frontend(port)
            p = None

        uvicorn.run(
            f"entry.web.app:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            workers=workers,
            **params,
        )
        if dev_mode and p is not None:
            p.terminate()
            p.wait()

    @staticmethod
    def _install_frontend():
        install_cmd = ["npm", "install"]
        subprocess.run(
            install_cmd,
            cwd=str(Path(__file__).parent / "../../../entry/web/view"),
        )

    def _build_frontend(self, port):
        web_dir = self._work_dir / ".web"
        if web_dir.exists():
            shutil.rmtree(web_dir)
        web_dir.mkdir(parents=True, exist_ok=True)

        build_cmd = ["npm", "run", "build"]
        build_env = os.environ.copy()
        build_env["VITE_PORT"] = str(port)
        build_env["VITE_OUTDIR"] = str(web_dir / "dist")  # Tell Vite where to output

        self._logger.info(f"Building frontend to {web_dir / 'dist'}")
        result = subprocess.run(
            build_cmd,
            env=build_env,
            cwd=str(Path(__file__).parent / "../../../entry/web/view"),
        )

        if result.returncode != 0:
            self._logger.error("Frontend build failed")
            return

        built_dir = web_dir / "dist"
        if built_dir.exists():
            self._logger.info(f"Frontend build completed successfully at {built_dir}")
        else:
            self._logger.error(
                f"Built directory {built_dir} does not exist after build"
            )
