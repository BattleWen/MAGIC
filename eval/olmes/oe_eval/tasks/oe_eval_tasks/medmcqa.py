"""
MedMCQA
"""

from oe_eval.tasks.base_task import MultipleChoiceTask
from oe_eval.tasks.utils import make_cloze_prompt, make_mcq_prompt

_CITATION = """
@InProceedings{pmlr-v174-pal22a,
  title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {248--260},
  year = 	 {2022},
  editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = 	 {174},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--08 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
  url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
  abstract = 	 {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS &amp; NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects &amp; topics. A detailed explanation of the solution, along with the above information, is provided in this study.}
}
"""


class MedMCQA(MultipleChoiceTask):
    VERSION = 0
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "openlifescienceai/medmcqa",
        "native_id_field": "id",
        "primary_metric": "acc_per_char",
        "split": "test",
    }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                self.dataset["train"].map(self._process_doc, with_indices=True)
            )
        return self._training_docs

    def validation_docs(self):
        return list(self.dataset["validation"].map(self._process_doc, with_indices=True))

    def test_docs(self):
        return list(self.dataset["test"].map(self._process_doc, with_indices=True))

    def unconditioned_prompt(self):
        return "Answer:"

    def _process_doc(self, doc, index=-1):
        query = make_cloze_prompt(doc["question"])
        out_doc = {
            "index": index,
            "query": query,
            "choices": [doc["opa"], doc["opb"], doc["opc"], doc["opd"]],
            "gold": int(doc["cop"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]


class MedMCQAMC(MedMCQA):
    TASK_CONFIG_DEFAULTS: dict = {
        "dataset_path": "openlifescienceai/medmcqa",
        "native_id_field": "id",
        "primary_metric": "acc_per_char",
        "split": "test",
    }

    def _process_doc(self, doc, index=-1):
        choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
        num_choices = len(choices)
        choice_labels = ["A", "B", "C", "D"][:num_choices]
        query = make_mcq_prompt(doc["question"], choices)
        out_doc = {
            "index": index,
            "query": query,
            "choices": choice_labels,
            "gold": int(doc["cop"]),
        }
        return out_doc

    def unconditioned_prompt(self):
        # Don't need unconditioned normalization here
        return None
