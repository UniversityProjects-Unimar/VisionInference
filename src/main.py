from src.config.settings import settings
from src.config.logger import LoggerClass as logger
from src.pipelines.inference_pipeline import InferencePipeline

def run():
    logger.configure(file_name="main", debug=True)
    pipeline = InferencePipeline()
    pipeline.run()


if __name__ == "__main__":
    run()