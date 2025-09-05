"""Experiment runner for evaluating RAG systems over datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from agentic_rag.agent.loop import AgenticRAGLoop, AgentResponse
from agentic_rag.eval.ragas_wrap import EvaluationResult, RAGASEvaluator
from agentic_rag.utils.jsonl import JSONLWriter
from agentic_rag.utils.timing import Timer


class ExperimentConfig(BaseModel):
    """Configuration for an evaluation experiment."""

    name: str
    description: str
    dataset_name: str
    model_config: Dict[str, Any]
    retrieval_config: Dict[str, Any]
    agent_config: Dict[str, Any]
    evaluation_config: Dict[str, Any] = {}
    output_dir: str = "./artifacts/experiments"
    max_samples: Optional[int] = None
    random_seed: int = 42


class ExperimentResult(BaseModel):
    """Result from a single experiment run."""

    config: ExperimentConfig
    query: str
    ground_truth: Optional[str] = None
    agent_response: AgentResponse
    evaluation: EvaluationResult
    timing: Dict[str, float]
    metadata: Dict[str, Any] = {}


class ExperimentSummary(BaseModel):
    """Summary statistics from an experiment."""

    config: ExperimentConfig
    total_samples: int
    successful_samples: int
    failed_samples: int
    avg_faithfulness: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_response_time: float
    avg_total_rounds: float
    avg_confidence: float
    metadata: Dict[str, Any] = {}


class ExperimentRunner:
    """Runner for conducting RAG evaluation experiments."""

    def __init__(
        self,
        agent: AgenticRAGLoop,
        evaluator: RAGASEvaluator,
        output_dir: Union[str, Path] = "./artifacts/experiments",
    ) -> None:
        """
        Initialize experiment runner.

        Args:
            agent: Agentic RAG agent to evaluate
            evaluator: Evaluation metrics calculator
            output_dir: Directory for experiment outputs
        """
        self.agent = agent
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_experiment(
        self,
        config: ExperimentConfig,
        dataset: List[Dict[str, Any]],
        save_results: bool = True,
    ) -> ExperimentSummary:
        """
        Run a complete evaluation experiment.

        Args:
            config: Experiment configuration
            dataset: List of dataset samples
            save_results: Whether to save results to disk

        Returns:
            Experiment summary with aggregate metrics
        """
        # TODO: Implement complete experiment execution
        raise NotImplementedError("Experiment execution not yet implemented")

    async def run_single_sample(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """
        Run evaluation on a single sample.

        Args:
            query: Input query
            ground_truth: Optional ground truth answer
            metadata: Optional additional metadata

        Returns:
            Single experiment result
        """
        timer = Timer()

        # Time the agent processing
        timer.start("agent_processing")
        try:
            agent_response = await self.agent.process_query(query)
            timer.stop("agent_processing")

            # Time the evaluation
            timer.start("evaluation")
            evaluation = self.evaluator.evaluate_response(
                query, agent_response, ground_truth
            )
            timer.stop("evaluation")

            # Create experiment result
            result = ExperimentResult(
                config=ExperimentConfig(
                    name="single_sample",
                    description="Single sample evaluation",
                    dataset_name="manual",
                    model_config={},
                    retrieval_config={},
                    agent_config={},
                ),
                query=query,
                ground_truth=ground_truth,
                agent_response=agent_response,
                evaluation=evaluation,
                timing=timer.get_times(),
                metadata=metadata or {},
            )

            return result

        except Exception as e:
            timer.stop("agent_processing")
            # Create failed result
            return ExperimentResult(
                config=ExperimentConfig(
                    name="single_sample",
                    description="Single sample evaluation",
                    dataset_name="manual",
                    model_config={},
                    retrieval_config={},
                    agent_config={},
                ),
                query=query,
                ground_truth=ground_truth,
                agent_response=AgentResponse(
                    query=query,
                    answer="Error occurred during processing",
                    confidence=0.0,
                    total_rounds=0,
                    contexts_used=[],
                    reasoning_trace=[f"Error: {str(e)}"],
                ),
                evaluation=EvaluationResult(
                    query=query,
                    answer="Error occurred",
                    contexts=[],
                ),
                timing=timer.get_times(),
                metadata={"error": str(e)},
            )

    def _save_results(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult],
        summary: ExperimentSummary,
    ) -> None:
        """
        Save experiment results to disk.

        Args:
            config: Experiment configuration
            results: List of individual results
            summary: Experiment summary
        """
        experiment_dir = self.output_dir / config.name
        experiment_dir.mkdir(exist_ok=True)

        # Save individual results
        results_file = experiment_dir / "results.jsonl"
        with JSONLWriter(results_file) as writer:
            for result in results:
                writer.write(result.model_dump())

        # Save summary
        summary_file = experiment_dir / "summary.json"
        with open(summary_file, "w") as f:
            import json

            json.dump(summary.model_dump(), f, indent=2)

        # Save config
        config_file = experiment_dir / "config.json"
        with open(config_file, "w") as f:
            import json

            json.dump(config.model_dump(), f, indent=2)

    def _calculate_summary(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult],
    ) -> ExperimentSummary:
        """
        Calculate summary statistics from experiment results.

        Args:
            config: Experiment configuration
            results: List of experiment results

        Returns:
            Experiment summary with aggregate metrics
        """
        successful_results = [r for r in results if "error" not in r.metadata]
        failed_results = [r for r in results if "error" in r.metadata]

        summary = ExperimentSummary(
            config=config,
            total_samples=len(results),
            successful_samples=len(successful_results),
            failed_samples=len(failed_results),
        )

        if successful_results:
            # Calculate average metrics
            faithfulness_scores = [
                r.evaluation.faithfulness
                for r in successful_results
                if r.evaluation.faithfulness is not None
            ]
            if faithfulness_scores:
                summary.avg_faithfulness = sum(faithfulness_scores) / len(
                    faithfulness_scores
                )

            # Similar calculations for other metrics
            precision_scores = [
                r.evaluation.context_precision
                for r in successful_results
                if r.evaluation.context_precision is not None
            ]
            if precision_scores:
                summary.avg_context_precision = sum(precision_scores) / len(
                    precision_scores
                )

            recall_scores = [
                r.evaluation.context_recall
                for r in successful_results
                if r.evaluation.context_recall is not None
            ]
            if recall_scores:
                summary.avg_context_recall = sum(recall_scores) / len(recall_scores)

            relevancy_scores = [
                r.evaluation.answer_relevancy
                for r in successful_results
                if r.evaluation.answer_relevancy is not None
            ]
            if relevancy_scores:
                summary.avg_answer_relevancy = sum(relevancy_scores) / len(
                    relevancy_scores
                )

            # Calculate timing and agent metrics
            response_times = [
                r.timing.get("agent_processing", 0.0) for r in successful_results
            ]
            summary.avg_response_time = sum(response_times) / len(response_times)

            total_rounds = [r.agent_response.total_rounds for r in successful_results]
            summary.avg_total_rounds = sum(total_rounds) / len(total_rounds)

            confidences = [r.agent_response.confidence for r in successful_results]
            summary.avg_confidence = sum(confidences) / len(confidences)

        return summary

    def load_experiment_results(
        self,
        experiment_name: str,
    ) -> tuple[ExperimentConfig, List[ExperimentResult], ExperimentSummary]:
        """
        Load previously saved experiment results.

        Args:
            experiment_name: Name of the experiment to load

        Returns:
            Tuple of (config, results, summary)
        """
        experiment_dir = self.output_dir / experiment_name

        # Load config
        config_file = experiment_dir / "config.json"
        with open(config_file) as f:
            import json

            config_data = json.load(f)
            config = ExperimentConfig(**config_data)

        # Load results
        results_file = experiment_dir / "results.jsonl"
        results = []
        with open(results_file) as f:
            for line in f:
                import json

                result_data = json.loads(line)
                results.append(ExperimentResult(**result_data))

        # Load summary
        summary_file = experiment_dir / "summary.json"
        with open(summary_file) as f:
            import json

            summary_data = json.load(f)
            summary = ExperimentSummary(**summary_data)

        return config, results, summary
