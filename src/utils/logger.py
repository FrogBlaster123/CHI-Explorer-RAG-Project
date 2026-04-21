import logging

class RAGLogger:
    def __init__(self, name="RAG_Logger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
    def info(self, msg):
        self.logger.info(msg)
        
    def log_retrieval_stage(self, stage: str, results: list):
        """Standardized logger for hybrid retrieval pipeline stages."""
        self.logger.info(f"=== {stage.upper()} ===")
        for idx, res in enumerate(results):
            score = res.get('score', 0.0)
            chunk_id = res.get('id', 'N/A')
            self.logger.info(f"[{idx+1}] ID: {chunk_id} | Score: {score:.4f}")
            
logger = RAGLogger()
