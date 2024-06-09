# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The BioCreative VI Chemical-Protein interaction dataset identifies entities of
chemicals and proteins and their likely relation to one other. Compounds are
generally agonists (activators) or antagonists (inhibitors) of proteins. The
script loads dataset in bigbio schema (using knowledgebase schema: schemas/kb)
AND/OR source (default) schema
"""
import datasets
import itertools

from bioc import pubtator
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = True
_LOCAL = False
_CITATION = """\
@InProceedings{zhang-etal:2022:LREC,
  author = {Dongxu Zhang and Sunil Mohan and Michaela Torkar and Andrew McCallum},
  title = {A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes},
  booktitle = {Proceedings of The 13th Language Resources and Evaluation Conference},
  month = {June},
  year = {2022},
  address = {Marseille, France},
  publisher = {European Language Resources Association},
}
"""
_DESCRIPTION = """\
The ChemDisGene dataset is a collection of biomedical research abstracts annotated with 
mentions of chemical, disease and gene/gene-product entities, and pairwise relationships 
between those entities.
"""

_DATASETNAME = "chem_dis_gene"
_DISPLAYNAME = "ChemDisGene"

_HOMEPAGE = "https://github.com/chanzuckerberg/ChemDisGene"

_LICENSE = 'CC0_1p0'

_URLs = {
    "curated": {
        "rel_ctd": "https://github.com/chanzuckerberg/ChemDisGene/raw/master/data/curated/approved_relns_ctd_v1.tsv.gz",
        "rel_new": "https://github.com/chanzuckerberg/ChemDisGene/raw/master/data/curated/approved_relns_new_v1.tsv.gz",
        "abstracts": "https://github.com/chanzuckerberg/ChemDisGene/raw/master/data/curated/abstracts.txt.gz"
    }
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION, Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ChemDisGeneDataset(datasets.GeneratorBasedBuilder):
    """ChemDisGene dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="chem_dis_gene_curated_bigbio_kb",
            version=BIGBIO_VERSION,
            description="ChemDisGene BigBio schema",
            schema="bigbio_kb",
            subset_id="chem_dis_gene_curated",
        ),
        BigBioConfig(
            name="chem_dis_gene_curated_chem_dis_bigbio_kb",
            version=BIGBIO_VERSION,
            description="ChemDisGene BigBio schema",
            schema="bigbio_kb",
            subset_id="chem_dis_gene_curated_chem_dis",
        ),
        BigBioConfig(
            name="chem_dis_gene_curated_chem_gene_bigbio_kb",
            version=BIGBIO_VERSION,
            description="ChemDisGene BigBio schema",
            schema="bigbio_kb",
            subset_id="chem_dis_gene_curated_chem_gene",
        ),
        BigBioConfig(
            name="chem_dis_gene_curated_gene_dis_bigbio_kb",
            version=BIGBIO_VERSION,
            description="ChemDisGene BigBio schema",
            schema="bigbio_kb",
            subset_id="chem_dis_gene_curated_gene_dis",
        ),
    ]

    DEFAULT_CONFIG_NAME = "chem_dis_gene_curated_bigbio_kb"

    def _info(self):

        if self.config.schema == "bigbio_kb":
            features = kb_features
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.subset_id.startswith("chem_dis_gene_curated"):
            urls = _URLs["curated"]
        else:
            raise NotImplementedError()

        data_files = {}
        for key, url in urls.items():
            data_files[key] = Path(dl_manager.download_and_extract(url))

        relation_type_prefix = None
        if self.config.subset_id == "chem_dis_gene_curated_chem_dis":
            relation_type_prefix = "chem_disease:"
        elif self.config.subset_id == "chem_dis_gene_curated_chem_gene":
            relation_type_prefix = "chem_gene:"
        elif self.config.subset_id == "chem_dis_gene_curated_gene_dis":
            relation_type_prefix = "gene_disease:"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": data_files,
                    "relation_type_prefix": relation_type_prefix
                },
            )
        ]

    def _generate_examples(
            self,
            data_files: Dict[str, Path],
            relation_type_prefix: Optional[str]
    ):
        if self.config.subset_id.startswith("chem_dis_gene_curated"):
            doc_to_relations = self._read_relations(
                relation_files=[data_files["rel_ctd"], data_files["rel_new"]],
                relation_type_prefix=relation_type_prefix
            )

            for example in self._pubtator_to_bigbio_kb(data_files["abstracts"]):
                db_id_to_entity_ids = defaultdict(list)
                for entity in example["entities"]:
                    for db_norm in entity["normalized"]:
                        entity_db_id = f"{db_norm['db_name']}:{db_norm['db_id']}"
                        db_id_to_entity_ids[entity_db_id].append(entity["id"])

                relations = doc_to_relations[example["id"]]

                bigbio_relations = []
                id_counter = itertools.count()

                for relation in relations:
                    relation_type = relation["type"]

                    arg1_entity_ids = db_id_to_entity_ids[relation["arg1"]]
                    arg2_entity_ids = db_id_to_entity_ids[relation["arg2"]]

                    bigbio_relations.extend([
                        {
                            "id": example["id"] + "_r_" + str(next(id_counter)),
                            "type": relation_type,
                            "arg1_id": arg1,
                            "arg2_id": arg2,
                            "normalized": []

                        }
                        for arg1 in arg1_entity_ids
                        for arg2 in arg2_entity_ids
                    ])

                example["relations"] = bigbio_relations

                yield example["id"], example

    @staticmethod
    def _read_relations(
            relation_files: List[Path],
            relation_type_prefix: Optional[str]
    ) -> Dict[str, List[Dict]]:
        relations = defaultdict(list)

        for relation_file in relation_files:
            with relation_file.open("r", encoding="utf8") as file:
                for line in file.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    pmid, relation_type, arg1, arg2 = line.split("\t")
                    if relation_type_prefix is not None:
                        if not relation_type.startswith(relation_type_prefix):
                            continue

                        relation_type = relation_type[len(relation_type_prefix):]

                    arg1 = arg1 if not arg1.isnumeric() else f"NCBI:{arg1}"
                    arg2 = arg2 if not arg2.isnumeric() else f"NCBI:{arg2}"

                    relations[pmid].append({
                        "type": relation_type,
                        "arg1": arg1,
                        "arg2": arg2
                    })

        return relations

    @staticmethod
    def _pubtator_to_bigbio_kb(abstracts_file: Path) -> Iterator[Dict]:
        with abstracts_file.open("r", encoding="utf8") as f:
            example = {}

            for doc in pubtator.iterparse(f):
                example["id"] = doc.pmid
                example["document_id"] = doc.pmid

                example["passages"] = [
                    {
                        "id": doc.pmid + "_title",
                        "type": "title",
                        "text": [doc.title],
                        "offsets": [[0, len(doc.title)]],
                    },
                    {
                        "id": doc.pmid + "_abstract",
                        "type": "abstract",
                        "text": [doc.abstract],
                        "offsets": [
                            [
                                # +1 assumes the title and abstract will be joined by a space.
                                len(doc.title) + 1,
                                len(doc.title) + 1 + len(doc.abstract),
                            ]
                        ],
                    },
                ]

                unified_entities = []
                for i, entity in enumerate(doc.annotations):
                    # We need a unique identifier for this entity, so build it from the document id and running counter
                    unified_entity_id = "_".join([doc.pmid, "e", str(i)])
                    normalized = []

                    for x in entity.id.split("|"):
                        if x == "-":
                            continue

                        low_x = x.lower()
                        if low_x.startswith("omim") or low_x.startswith("mesh"):
                            db_name, db_id = x.strip().split(":")
                            normalized.append({"db_name": db_name, "db_id": db_id})
                        elif x.isnumeric():
                            normalized.append({"db_name": "NCBI", "db_id": x})
                        else:
                            raise AssertionError("The database id should either be a MESH/OMIM or a NCBI gene id!")

                    unified_entities.append(
                        {
                            "id": unified_entity_id,
                            "type": entity.type,
                            "text": [entity.text],
                            "offsets": [[entity.start, entity.end]],
                            "normalized": normalized,
                        }
                    )

                example["entities"] = unified_entities
                example["relations"] = []
                example["events"] = []
                example["coreferences"] = []

                yield example
