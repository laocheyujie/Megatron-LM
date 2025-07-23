# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass
class MaskedWordPieceDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core Masked WordPiece datasets"""

    masking_probability: float = None
    """The probability we mask a candidate N-gram"""

    short_sequence_probability: float = None
    """The probability we return a sequence shorter than the target sequence length"""

    masking_max_ngram: int = None
    """The maximum length N-gram to consider masking or permuting"""

    masking_do_full_word: bool = None
    """Whether we mask the whole word or its component parts"""

    masking_do_permutation: bool = None
    """Whether we shuffle a subset of candidate N-grams in addition"""

    masking_use_longer_ngrams: bool = None
    """Whether to favor longer N-grams over shorter N-grams"""

    masking_use_geometric_distribution: bool = None
    """Whether to draw the size of the N-gram from a geometric distribution according to SpanBERT
       https://arxiv.org/abs/1907.10529 (Section 3.1)
    """

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.masking_probability is not None
        assert self.short_sequence_probability is not None
        assert self.masking_max_ngram is not None
        assert self.masking_do_full_word is not None
        assert self.masking_do_permutation is not None
        assert self.masking_use_longer_ngrams is not None
        assert self.masking_use_geometric_distribution is not None

        assert self.masking_probability > 0 and self.masking_probability < 1.0
        assert self.short_sequence_probability >= 0 and self.short_sequence_probability <= 1.0
        assert self.masking_max_ngram > 0
        assert not (self.masking_use_geometric_distribution and self.masking_do_permutation)

        if self.masking_use_geometric_distribution and self.masking_use_longer_ngrams:
            log_single_rank(
                logger,
                logging.WARNING,
                "The use of a geometric distribution overrides the default distribution",
            )


class MaskedWordPieceDataset(MegatronDataset):
    """The semi-abstract base class for masked WordPiece datasets

    This implementation makes the rigid assumption that all inheritor datasets are built upon the
    IndexedDataset class. This assumption may be pushed down to the inheritors in future if
    necessary.

    NB: WordPiece tokenization prepends a double hash "##" to all tokens/pieces in a word, save the
    first token/piece.

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the
                                          MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset.
                                     When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (MaskedWordPieceDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: MaskedWordPieceDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        return low_level_dataset.document_indices.shape[0] - 1

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: MaskedWordPieceDatasetConfig
    ) -> IndexedDataset:
        return IndexedDataset(dataset_path)

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        Returns:
            List[str]: The key config attributes
        """
        return super(MaskedWordPieceDataset, MaskedWordPieceDataset)._key_config_attributes() + [
            "masking_probability",
            "short_sequence_probability",
            "masking_max_ngram",
            "masking_do_full_word",
            "masking_do_permutation",
            "masking_use_longer_ngrams",
            "masking_use_geometric_distribution",
        ]

    def __len__(self) -> int:
        return self.sample_index.shape[0]

    def _build_sample_index(
        self, sequence_length: int, min_sentences_per_sample: int
    ) -> numpy.ndarray:
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_description = get_path_to("description.txt")
        path_to_sample_index = get_path_to("sample_index.npy")
        cache_hit = all(map(os.path.isfile, [path_to_description, path_to_sample_index]))

        if self.num_samples is not None:
            num_epochs = numpy.iinfo(numpy.int32).max - 1
        else:
            num_epochs = 1

        if not cache_hit and torch.distributed.get_rank() == 0:
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the sample index
            log_single_rank(
                logger,
                logging.INFO,
                f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
            )
            t_beg = time.time()
            from megatron.core.datasets import helpers

            # Add +1 for access to document upper bound
            indices = numpy.append(self.indices, self.indices[-1] + 1)

            sample_index = helpers.build_mapping(
                self.dataset.document_indices[indices],
                self.dataset.sequence_lengths,
                num_epochs,
                self.num_samples,
                sequence_length,
                self.config.short_sequence_probability,
                self.config.random_seed,
                False,
                min_sentences_per_sample,
            )
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0]}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return sample_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return sample_index

    def _create_masked_lm_predictions(
        self,
        token_ids: List[int],
        target_sequence_length: int,
        numpy_random_state: numpy.random.RandomState,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[Tuple[List[int], List[int]]]]:
        """Creates the predictions for the masked LM objective

        Args:
            token_ids (List[int]): The token ids
            target_sequence_length (int): The target sequence length
            numpy_random_state (numpy.random.RandomState): The NumPy random state

        Returns:
            Tuple[List[int], List[int], List[int], List[int], List[Tuple[List[int], List[int]]]]:
                1. masked_token_ids -> The masked sequence
                2. masked_positions -> The indices for the masked token ids
                3. masked_labels    -> The original token ids for the masked token ids
                4. boundaries       -> The sentence and word boundaries for the sequence
                4. masked_spans     -> The masked positions and labels with N-gram info intact
        """
        # Build the token sentence and word boundaries and the masking candidates
        # e.g. [cls, id, ##id, ##id, id, ##id, sep, id, ##id, sep]
        #    -> boundaries: [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        #    -> candidates with whole word masking: [[1, 2, 3], [4, 5], [7, 8]]
        #    -> candidates sans whole word masking: [[1], [2], [3], [4], [5], [7], [8]]
        boundaries = []
        candidates = []
        for i, token_id in enumerate(token_ids):
            if token_id == self.config.tokenizer.cls or token_id == self.config.tokenizer.sep:
                boundaries.append(1)
            else:
                if not self.config.tokenizer.inv_vocab[token_id].startswith("##"):
                    # NOTE: 当前的 sub-word 是否是一个 word 的开始，如果是，则 boundary[i] = 1
                    boundaries.append(1)
                    candidates.append([i])
                else:
                    boundaries.append(0)
                    if self.config.masking_do_full_word and len(candidates) > 0:
                        # NOTE: 非开头 sub-word 追加到已有的开头 sub-word 上
                        candidates[-1].append(i)
                    else:
                        candidates.append([i])
        '''
        candidates
        [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], 
            [13, 14, 15], [16], [17], [18], [19], [20, 21], 
            [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], 
            [34], [35], [36], [37], [38], 
            [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], 
            [50], [51], [52], [53], 
            [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], 
            [65], [66], [67], [68], 
            [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], 
            [80], [81], [82], [83, 84], 
            [85], [86], [87, 88, 89], [90], [91], [92], [93], [94], [95], [96], 
            [97], [98], [99], [100, 101, 102], 
            [103], [104], [105], [106], [107, 108], [109], [110], [111], [112], 
            [113], [114], [115], [116], [117], 
            ...
        ]
        '''

        n_maskings = min(
            self.config.masking_probability * target_sequence_length,
            max(1, int(round(len(token_ids) * self.config.masking_probability))),
        )

        ngram_nvals = numpy.arange(self.config.masking_max_ngram, dtype=numpy.int64) + 1

        # By default, the N-gram probabilities are inversely proportional to N
        # e.g. N = 3
        #    -> P = array([0.54545455, 0.27272727, 0.18181818])
        nprobs = 1.0 / ngram_nvals
        nprobs = nprobs / nprobs.sum(keepdims=True)
        if self.config.masking_use_longer_ngrams:
            nprobs = nprobs[::-1]

        # Create a nested list of depth 3
        #   layer 1: the candidate dimension
        #   layer 2: the N-gram dimension
        #   layer 3: the token dimension
        # NOTE: idx 都是打头词 sub-word，不以 ## 开头
        candidate_ngrams = [
            [candidates[idx : idx + n] for n in ngram_nvals] for idx in range(len(candidates))
        ]
        numpy_random_state.shuffle(candidate_ngrams)
        '''
        candidate_ngrams
        [[[[0]], 
        [[0], [1]], 
        [[0], [1], [2]], 
        [[0], [1], [2], [3]], 
        [[0], [1], [2], [3], [4]], 
        [[0], [1], [2], [3], [4], [5]], 
        [[0], [1], [2], [3], [4], [5], [6]], 
        [[0], [1], [2], [3], [4], [5], [6], [7]], 
        [[0], [1], [2], [3], [4], [5], [6], [7], [8]], 
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]], 
        [[[1]], 
        [[1], [2]], 
        [[1], [2], [3]],...
        '''

        masked_token_ids = list(token_ids)
        masked_positions_and_labels = []
        masked_spans = []
        masked_indices = set()
        for candidate_idx in range(len(candidate_ngrams)):
            '''
            [[[57]], 
            [[57], [58]], [[57], [58], [59]], [[57], [58], [59], [60]], 
            [[57], [58], [59], [60], [61]], 
            [[57], [58], [59], [60], [61], [62]], 
            [[57], [58], [59], [60], [61], [62], [63]], 
            [[57], [58], [59], [60], [61], [62], [63], [64]],
            [[57], [58], [59], [60], [61], [62], [63], [64], [65]], 
            [[57], [58], [59], [60], [61], [62], [63], [64], [65], [66]]]
            '''
            n_ngrams = len(candidate_ngrams[candidate_idx])

            # Stop when we hit our desired number of maskings
            if len(masked_positions_and_labels) >= n_maskings:
                break

            # Do nothing for candidates with no ngrams
            if not candidate_ngrams[candidate_idx]:
                continue

            # Choose the initial value of N
            if self.config.masking_use_geometric_distribution:
                # Sample N from a geometric distribution with p = 0.2 and clip
                # i.e. SpanBERT
                #    -> https://arxiv.org/abs/1907.10529 (Section 3.1)
                p = 0.2
                n = min(numpy_random_state.geometric(p), self.config.masking_max_ngram)
            else:
                p = nprobs[:n_ngrams] / nprobs[:n_ngrams].sum(keepdims=True)
                n = numpy_random_state.choice(ngram_nvals[:n_ngrams], p=p)

            while True:
                ngram_indices = sum(candidate_ngrams[candidate_idx][n - 1], [])
                n = n - 1
                # Success: masking this N-gram puts us below the desired number of maskings
                if n_maskings >= len(masked_positions_and_labels) + len(ngram_indices):
                    skip_candidate = False
                    break
                # Failure: no N-grams remain for this candidate
                if n == 0:
                    skip_candidate = True
                    break

            # Do nothing for candidates whose 1-gram is too long
            if skip_candidate:
                continue

            # Do nothing for candidate indices which have already been masked
            if any(map(lambda idx: idx in masked_indices, ngram_indices)):
                continue

            # Mask the tokens and record their original positions and values
            for index in ngram_indices:
                masked_indices.add(index)
                mask = self._get_token_mask(numpy_random_state)
                if mask is None:
                    masked_token_ids[index] = token_ids[index]
                else:
                    # NOTE: masked_token_ids 的第 index 位置被赋值为 mask (103)
                    masked_token_ids[index] = mask
                # NOTE: 记录 masked_token_ids 的第 index 位置和对应的 token id
                masked_positions_and_labels.append((index, token_ids[index]))

            # NOTE: ngram_indices: 索引 list; [token_ids[index] for index in ngram_indices]: 索引对应的 token id
            # NOTE: len(ngram_indices) 表示 mask 的 n-gram 中的 n
            masked_spans.append((ngram_indices, [token_ids[index] for index in ngram_indices]))
            '''
            [
                ([57, 58], [126, 110]),
                ([61], [117]),
                ([28], [385]),
                ([184, 185, 186], [22100, 2690, 114]),
                ([147], [10351]),
                ([158, 159, 160, 161, 162, 163, 164, 165, 166, 167], [1157, 24647, 1111, 10351, 15503, 1120, 127, 119, 124, 110]),
                ([32, 33, 34], [2199, 2603, 1106])
            '''

        assert len(masked_positions_and_labels) <= n_maskings

        numpy_random_state.shuffle(candidate_ngrams)

        if self.config.masking_do_permutation:
            n_swappings = n_maskings

            permuted_indices = set()
            for candidate_idx in range(len(candidate_ngrams)):
                n_ngrams = len(candidate_ngrams[candidate_idx])

                if len(permuted_indices) >= n_swappings:
                    break

                # Do nothing for candidates with no ngrams
                if not candidate_ngrams[candidate_idx]:
                    continue

                p = nprobs[:n_ngrams] / nprobs[:n_ngrams].sum(keepdims=True)
                n = numpy.random.choice(ngram_nvals[:n_ngrams], p=p)

                while True:
                    ngram_indices = sum(candidate_ngrams[candidate_idx][n - 1], [])
                    n = n - 1
                    # Success: swapping this N-gram puts us below the desired number of swappings
                    if n_swappings >= len(permuted_indices) + len(ngram_indices):
                        skip_candidate = False
                        break
                    # Failure: no N-grams remain for this candidate
                    if n == 0:
                        skip_candidate = True
                        break

                # Do nothing for candidates whose 1-gram is too long
                if skip_candidate:
                    continue

                # Do nothing for candidate indices which have already been masked or permuted
                if any(
                    map(lambda idx: idx in masked_indices or idx in permuted_indices, ngram_indices)
                ):
                    continue

                for index in ngram_indices:
                    permuted_indices.add(index)

            assert len(permuted_indices) <= n_swappings

            permuted_indices = sorted(permuted_indices)
            permuted_indices_copy = list(permuted_indices)
            numpy_random_state.shuffle(permuted_indices_copy)
            masked_token_ids_copy = list(masked_token_ids)

            for idx, idx_copy in zip(permuted_indices, permuted_indices_copy):
                masked_token_ids[idx] = masked_token_ids_copy[idx_copy]
                masked_positions_and_labels.append((idx, masked_token_ids_copy[idx]))

        masked_positions_and_labels = sorted(masked_positions_and_labels, key=lambda x: x[0])
        masked_positions = []
        masked_labels = []
        for position, label in masked_positions_and_labels:
            masked_positions.append(position)
            masked_labels.append(label)

        masked_spans = sorted(masked_spans, key=lambda x: x[0][0])

        return (masked_token_ids, masked_positions, masked_labels, boundaries, masked_spans)
        '''
        masked_token_ids:
        [5231, 2129, 3085, 7484, 1514, 2199, 2603, 1106, 1542, 110, 131, 4195, 
        119, 148, 17444, 2559, 117, 1351, 
        1407, 113, 11336, 27603, 114, 118, 5231, 787, 188, 2129, 103, 2195, 
        1157, 1514, 103, 103, 103, 1542, 110, 
        1121, 1542, 119, 126, 110, 1113, 9170, 117, 1229, 5920, 1157, 2670, 
        3213, 24647, 1106, 124, 110, 1121, 123, 
        119, 103, 103, 1111, 10351, 103, 1122, 1163, 1107, 170, 4195, 119, 
        1135, 1145, 2023, 1157, 24647, 1111, 10351, 
        15503, 1120, 103, 103, 103, 103, 119, 113, 7178, 1158, 1118, 24130, 
        163, 27411, 1116, 132, 2269, 1118, 19317, 
        2902, 25960, 1118, 22100, 2690, 114, 148, 17444, 2559, 117, 1351, 
        1407, 113, 11336, 27603, 114, 118, 103, 103, 
        188, 2129, 3085, 2195, 1157, 1514, 2199, 2603, 1106, 1542, 110, 
        1121, 1542, 119, 126, 110, 1113, 9170, 117,
        1229, 5920, 1157, 2670, 3213, 24647, 1106, 124, 110, 1121, 123, 
        103, 126, 110, 1111, 103, 117, 1122, 1163, 
        1107, 170, 4195, 119, 1135, 1145, 2023, 103, 103, 103, 103, 103, 
        103, 103, 103, 103, 103, 119, 113, 7178, 1158, 
        1118, 24130, 163, 27411, 1116, 132, 2269, 1118, 19317, 2902, 25960, 
        1118, 103, 103, 103]
        
        其中 103 = mask-id
        
        masked_positions:
        [28, 32, 33, 34, 57, 58, 61, 77, 78, 79, 80, 111, 112, 143, 147, 158, 159, 
        160, 161, 162, 163, 164, 165, 166, 167, 184, 185, 186]
        
        masked_labels:
        [3085, 2199, 2603, 1106, 126, 110, 117, 127, 119, 124, 110, 5231, 787, 
        119, 10351, 1157, 24647, 1111, 10351, 15503, 1120, 127, 119, 124, 110, 22100, 2690, 114]
        
        boundaries:
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 
        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        masked_spans:
        [([28], [3085]), 
        ([32, 33, 34], [2199, 2603, 1106]), 
        ([57, 58], [126, 110]), 
        ([61], [117]), 
        ([77, 78, 79, 80], [127, 119, 124, 110]), 
        ([111, 112], [5231, 787]), ([143], [119]), ([147], [10351]), 
        ([158, 159, 160, 161, 162, 163, 164, 165, 166, 167], [1157, 24647, 1111, 10351, 15503, 1120, 127, 119, 124, 110]), 
        ([184, 185, 186], [22100, 2690, 114])]
        '''

    @abstractmethod
    def _get_token_mask(self, numpy_random_state: numpy.random.RandomState) -> Optional[int]:
        pass
