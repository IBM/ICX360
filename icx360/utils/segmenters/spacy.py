"""
Class and functions for segmenting input text into units using a spaCy model.

SpaCySegmenter is the main class.
The remaining functions implement an algorithm for segmentation into phrases.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

import numpy as np
import spacy


class SpaCySegmenter:
    """
    Class for segmenting input text into units using a spaCy model.

    Attributes:
        model (spacy.Language): spaCy model.
    """
    def __init__(self, spacy_model):
        """
        Initialize SpaCySegmenter object.

        Args:
            spacy_model (str): Name of spaCy model.
        """
        self.model = spacy.load(spacy_model)

    def segment_units(self, input_text, ind_segment=True, unit_types="s", sent_idxs=None, segment_type="w", max_phrase_length=10):
        """
        (Further) Segment input text into units.

        Args:
            input_text (str or list[str]):
                Input text as a single unit (if str) or existing sequence of units (list[str]).
            ind_segment (bool or list[bool]):
                Whether to segment entire input text or each existing unit.
                If bool, applies to all units. If list[bool], applies to each unit individually.
            unit_types (str or list[str]):
                Types of units in input_text:
                    "p" for paragraph, "s" for sentence, "w" for word,
                    "n" for not to be perturbed or segmented (fixed).
                If str, applies to all units in input_text, otherwise unit-specific.
            sent_idxs (list[int] or None):
                Index of sentence (or larger unit) that contains each existing unit.
            segment_type (str):
                Type of units to segment into: "s" for sentences, "w" for words, "ph" for phrases.
            max_phrase_length (int, optional):
                Maximum phrase length in terms of spaCy tokens.

        Returns:
            units (list[str]):
                Resulting sequence of units.
            unit_types (list[str]):
                Types of units.
            sent_idxs_new (list[int]):
                Index of sentence (or larger unit) that contains each unit.
        """
        if type(input_text) is str:
            # Convert to list of single unit
            input_text = [input_text]
        num_units_in = len(input_text)
        if type(ind_segment) is bool:
            ind_segment = [ind_segment]
        if type(unit_types) is str:
            unit_types = [unit_types] * num_units_in
        if sent_idxs is None:
            sent_idxs = list(range(num_units_in))

        units = []
        subunit_types = []
        sent_idxs_new = []
        sent_idx = 0

        # Iterate over existing units to be segmented
        for u, unit in enumerate(input_text):
            # Compatibility between unit type and segment type
            if segment_type.startswith("s"):
                unit_compat = unit_types[u] == "p"
            elif segment_type.startswith("ph"):
                unit_compat = unit_types[u] in ("p", "s")
            elif segment_type.startswith("w"):
                unit_compat = unit_types[u] not in ("w", "n")
            else:
                unit_compat = False

            if ind_segment[u] and unit_compat:

                # Segment unit into words and sentences using spaCy model
                doc = self.model(unit)

                # Iterate over subunits
                for subunit in doc if segment_type.startswith("w") else doc.sents:
                    if segment_type.startswith("ph"):
                        # Segment sentence into phrases
                        phrases, phrase_types = segment_into_phrases(subunit, doc, max_phrase_length=max_phrase_length)
                        # Append phrases and phrase types
                        units.extend([phrase.text_with_ws for phrase in phrases])
                        subunit_types.extend(phrase_types)
                        # Append sentence indices
                        sent_idxs_new.extend([sent_idx] * len(phrases))

                    else:
                        # Append subunit (word or sentence)
                        units.append(subunit.text_with_ws)
                        # Subunit type is segment type
                        subunit_types.append(segment_type[0])
                        # Append sentence index
                        sent_idxs_new.append(sent_idx)

                    if not segment_type.startswith("w"):
                        # Increment sentence index
                        sent_idx += 1

                if segment_type.startswith("w") and u < num_units_in - 1:
                    # Increment sentence index unless next unit belongs to same sentence
                    sent_idx += sent_idxs[u + 1] - sent_idxs[u]

            else:
                # Append unit
                units.append(unit)
                # Append unit type
                subunit_types.append(unit_types[u])
                # Append sentence index
                sent_idxs_new.append(sent_idx)
                if u < num_units_in - 1:
                    # Increment sentence index unless next unit belongs to same sentence
                    sent_idx += sent_idxs[u + 1] - sent_idxs[u]

        return units, subunit_types, sent_idxs_new


def segment_into_phrases(sent, doc, max_phrase_length=10):
    """
    Segment sentence (or span within sentence) into phrases.

    Args:
        sent (spacy.tokens.Span):
            Sentence or span to be segmented.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.
        max_phrase_length (int, optional):
            Maximum phrase length in terms of spaCy tokens.

    Returns:
        phrases (list[spacy.tokens.Span]):
            List of segmented phrases.
        phrase_types (list[str]):
            Types of phrases (e.g., "ROOT", "non-leaf", spaCy dependency labels).
    """
    # TODO: decrease max_phrase_length from default depending on len(sent)

    # Root word of sentence
    root = sent.root
    if root.is_space:
        # Handle pathological case where root is a space
        # It should have only one child, which is the actual sentence
        if root.n_lefts + root.n_rights != 1:
            # In case it has more than one child, take only the longest child
            max_child_length = 0
            for child in root.children:
                if child.right_edge.i + 1 - child.left_edge.i > max_child_length:
                    sent = doc[child.left_edge.i : child.right_edge.i + 1]
                    max_child_length = child.right_edge.i + 1 - child.left_edge.i
        else:
            for child in root.children:
                sent = doc[child.left_edge.i : child.right_edge.i + 1]
        # Re-process sentence as its own doc
        doc = sent.as_doc()
        sent = list(doc.sents)[0]
        root = sent.root

    # Initialize
    phrases = []
    phrase_types = []

    # Iterate over left children of root
    phrases, phrase_types, need_sort_left = append_or_segment_children(root.lefts, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)

    # Root span
    span = doc[root.i : root.i + 1]
    phrases.append(span)
    if root.dep_ == "ROOT":
        phrase_types.append("ROOT")
    else:
        phrase_types.append("non-leaf")

    # Iterate over right children of root
    phrases, phrase_types, need_sort_right = append_or_segment_children(root.rights, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)

    # Sort phrases if needed
    if need_sort_left or need_sort_right:
        phrases, phrase_types = sort_phrases(phrases, phrase_types)

    # Merge phrases that constitute a noun chunk
    phrases, phrase_types = merge_noun_chunk_phrases(phrases, phrase_types, sent.noun_chunks, doc)

    # Merge single-token phrases with neighbors
    phrases, phrase_types = merge_singleton_phrases(phrases, phrase_types, doc, max_phrase_length=max_phrase_length)

    return phrases, phrase_types

def append_or_segment_children(children, phrases, phrase_types, doc, max_phrase_length=10):
    """
    Append syntactic children of a node as phrases or further segment them.

    Args:
        children (generator[spacy.tokens.Token]):
            Generator of syntactic children.
        phrases (list[spacy.tokens.Span]):
            List of current phrases.
        phrase_types (list[str]):
            List of current phrase types.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.
        max_phrase_length (int):
            Maximum phrase length in terms of spaCy tokens.

    Returns:
        phrases (list[spacy.tokens.Span]):
            Updated list of phrases.
        phrase_types (list[str]):
            Updated list of phrase types.
        need_sort (bool):
            Flag to indicate whether phrases need sorting.
    """
    need_sort = False
    # Iterate over children
    for child in children:
        # Size and edges of child's subtree
        size_subtree = len(list(child.subtree))
        i_left_edge, i_right_edge = child.left_edge.i, child.right_edge.i

        if size_subtree == (i_right_edge - i_left_edge + 1):
            # Subtree is a contiguous span
            span = doc[i_left_edge : i_right_edge + 1]
            # Append or further segment span
            phrases, phrase_types = append_or_segment_span(span, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)

        else:
            print(f"Subtree {[token.text for token in child.subtree]} not contiguous!")
            need_sort = True

            # Iterate over tokens in subtree to find contiguous spans
            i_span_start, i_span_prev = i_left_edge, i_left_edge
            for token in child.subtree:
                if token.i == i_left_edge or token.i == i_span_prev + 1:
                    # First token or continuation of span
                    i_span_prev = token.i
                else:
                    # Span has ended
                    span = doc[i_span_start : i_span_prev + 1]
                    # Root tokens in span
                    span_roots = [token for token in span if token.head not in span]
                    if len(span_roots) > 1:
                        # Span contains multiple subtrees, re-process as its own doc
                        doc_span = span.as_doc()
                        span = list(doc_span.sents)[0]
                        phrases, phrase_types = append_or_segment_span(span, phrases, phrase_types, doc_span, max_phrase_length=max_phrase_length)
                        # Span contains multiple subtrees, treat them as separate children
                        # phrases, phrase_types, _ = append_or_segment_children(span_roots, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)
                    else:
                        # Append or further segment span
                        phrases, phrase_types = append_or_segment_span(span, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)
                    # Start new span
                    i_span_start, i_span_prev = token.i, token.i
            # Last span
            span = doc[i_span_start : i_span_prev + 1]
            # Root tokens in span
            span_roots = [token for token in span if token.head not in span]
            if len(span_roots) > 1:
                # Span contains multiple subtrees, re-process as its own doc
                doc_span = span.as_doc()
                span = list(doc_span.sents)[0]
                phrases, phrase_types = append_or_segment_span(span, phrases, phrase_types, doc_span, max_phrase_length=max_phrase_length)
                # Span contains multiple subtrees, treat them as separate children
                # phrases, phrase_types, _ = append_or_segment_children(span_roots, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)
            else:
                # Append or further segment span
                phrases, phrase_types = append_or_segment_span(span, phrases, phrase_types, doc, max_phrase_length=max_phrase_length)

    return phrases, phrase_types, need_sort

def append_or_segment_span(span, phrases, phrase_types, doc, max_phrase_length=10):
    """
    Append span to list of phrases or further segment span.

    Args:
        span (spacy.tokens.Span):
            Span to be appended or further segmented.
        phrases (list[spacy.tokens.Span]):
            List of current phrases.
        phrase_types (list[str]):
            List of current phrase types.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.
        max_phrase_length (int):
            Maximum phrase length in terms of spaCy tokens.

    Returns:
        phrases (list[spacy.tokens.Span]):
            Updated list of phrases.
        phrase_types (list[str]):
            Updated list of phrase types.
    """
    # Length of span in terms of non-punctuation non-space tokens
    num_nonpunct = sum( is_not_punct_space(span))
    if num_nonpunct > max_phrase_length:
        # Span too long, recursively segment into phrases
        subphrases, subphrase_types = segment_into_phrases(span, doc, max_phrase_length=max_phrase_length)
        phrases.extend(subphrases)
        phrase_types.extend(subphrase_types)
    else:
        # Append span as phrase
        phrases.append(span)
        # For a leaf phrase, its type is the dependency label of the phrase root
        phrase_types.append(span.root.dep_)

    return phrases, phrase_types

def is_not_punct_space(span):
    """
    Checks whether each token of span is not punctuation and not a space

    Args:
        span (spacy.tokens.Span)

    Returns:
        A list of Booleans where each element is True iff corresponding token is not punctuation and not a space.
    """
    return [(not token.is_punct) and (not token.is_space) for token in span]

def sort_phrases(phrases, phrase_types):
    """
    Sort phrases by their starting token index.

    Args:
        phrases (list[spacy.tokens.Span]):
            List of phrases.
        phrase_types (list[str]):
            List of phrase types.

    Returns:
        phrases (list[spacy.tokens.Span]):
            Sorted list of phrases.
        phrase_types (list[str]):
            Types of sorted phrases.
    """
    # Sort phrases by starting indices
    starts = np.array([phrase.start for phrase in phrases])
    idx_sort = starts.argsort()
    phrases = [phrases[idx] for idx in idx_sort]
    phrase_types = [phrase_types[idx] for idx in idx_sort]

    return phrases, phrase_types

def merge_noun_chunk_phrases(phrases, phrase_types, noun_chunks, doc):
    """
    Merge phrases that constitute a noun chunk.

    Args:
        phrases (list[spacy.tokens.Span]):
            List of phrases.
        phrase_types (list[str]):
            List of phrase types.
        noun_chunks (generator[spacy.tokens.Span]):
            Generator of noun chunks.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.

    Returns:
        phrases_merged (list[spacy.tokens.Span]):
            List of merged phrases.
        phrase_types_merged (list[str]):
            Types of merged phrases.
    """
    # Phrase boundaries
    starts_phrases = np.array([phrase.start for phrase in phrases])
    ends_phrases = np.array([phrase.end for phrase in phrases])

    spans_merge = []
    # Iterate over noun chunks
    for chunk in noun_chunks:
        # Phrase where noun chunk starts
        idx_start = np.nonzero(chunk.start >= starts_phrases)[0][-1]
        # Phrase where noun chunk ends
        try:
            idx_end = np.nonzero(chunk.end <= ends_phrases)[0][0]
        except IndexError:
            # Can't find phrase where noun chunk ends for some reason, skip
            continue

        if idx_end > idx_start:
            # Noun chunk spans multiple phrases
            # Check that noun chunk coincides with span of these phrases
            if not (chunk.start == starts_phrases[idx_start] and chunk.end == ends_phrases[idx_end]):
                continue
            # Find phrase containing root of noun chunk
            for idx_root in range(idx_start, idx_end + 1):
                if chunk.root in phrases[idx_root]:
                    break
            # Check that root phrase is a singleton
            if len(phrases[idx_root]) != 1:
                continue
            # Check that other phrases are children of root phrase
            for idx in range(idx_start, idx_end + 1):
                if not (idx == idx_root or phrases[idx].root.head == phrases[idx_root][0]):
                    continue

            # Mark phrase span for merging
            spans_merge.append((idx_start, idx_end))

    # Merge phrase spans
    phrases_merged, phrase_types_merged = merge_phrase_spans(phrases, phrase_types, spans_merge, doc)

    return phrases_merged, phrase_types_merged

def merge_phrase_spans(phrases, phrase_types, spans_merge, doc):
    """
    Merge phrases within specified spans of phrases.

    Args:
        phrases (list[spacy.tokens.Span]):
            List of phrases.
        phrase_types (list[str]):
            List of phrase types.
        spans_merge (list[tuple]):
            List of phrase spans, each a 2-element tuple of a starting phrase index and an ending phrase index.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.

    Returns:
        phrases_merged (list[spacy.tokens.Span]):
            List of merged phrases.
        phrase_types_merged (list[str]):
            Types of merged phrases.
    """
    # Initialize merged phrases with originals up to first phrase span to merge
    idx_end = spans_merge[0][0] if spans_merge else None
    phrases_merged = phrases[:idx_end]
    phrase_types_merged = phrase_types[:idx_end]

    # Iterate over phrase spans to merge
    for s, span_merge in enumerate(spans_merge):
        # Append merged phrase
        span = doc[phrases[span_merge[0]].start : phrases[span_merge[1]].end]
        phrases_merged.append(span)
        if span.n_lefts == 0 and span.n_rights == 0:
            # Merged phrase is a leaf, take type to be dependency label of root
            phrase_type = span.root.dep_
        else:
            # Merged phrase is non-leaf
            phrase_type = "non-leaf"
        phrase_types_merged.append(phrase_type)

        # Append original phrases up to next phrase span to merge
        idx_end = None if s == len(spans_merge) - 1 else spans_merge[s + 1][0]
        phrases_merged.extend(phrases[span_merge[1] + 1 : idx_end])
        phrase_types_merged.extend(phrase_types[span_merge[1] + 1 : idx_end])

    return phrases_merged, phrase_types_merged

def merge_singleton_phrases(phrases, phrase_types, doc, max_phrase_length=10):
    """
    Merge single-token phrases with their neighbors.

    Args:
        phrases (list[spacy.tokens.Span]):
            List of phrases.
        phrase_types (list[str]):
            List of phrase types.
        doc (spacy.tokens.Doc):
            spaCy Doc containing the sentence.
        max_phrase_length (int):
            Maximum phrase length in terms of spaCy tokens.

    Returns:
        phrases_merged (list[spacy.tokens.Span]):
            List of merged phrases.
        phrase_types_merged (list[str]):
            Types of merged phrases.
    """
    num_phrases = len(phrases)

    spans_merge = []
    skip_until = 0
    # Iterate over phrases
    for p, phrase in enumerate(phrases):
        if p < skip_until:
            # Skip phrase already marked for merging
            continue
        skip_until = p + 1

        if len(phrase) == 1 and (phrase_types[p] == "non-leaf" or phrase.root.dep_ == 'cc'):
            # Phrase is a singleton and either a non-root non-leaf or a coordinating conjunction

            # Check neighboring phrases to the left
            offset_l = 0
            remaining_length = max_phrase_length - (phrase.end - phrases[p - offset_l].start)
            while offset_l < p and merge_nbor_of_singleton_phrase(phrases[p - offset_l - 1], phrase, offset_l + 1, remaining_length):
                # Neighboring phrase meets criteria, check next phrase to the left
                offset_l += 1
                remaining_length = max_phrase_length - (phrase.end - phrases[p - offset_l].start)

            # Check neighboring phrases to the right
            offset_r = 0
            remaining_length = max_phrase_length - (phrases[p + offset_r].end - phrases[p - offset_l].start)
            while offset_r < num_phrases - 1 - p and merge_nbor_of_singleton_phrase(phrases[p + offset_r + 1], phrase, offset_r + 1, remaining_length):
                # Neighboring phrase meets criteria, check next phrase to the right
                offset_r += 1
                remaining_length = max_phrase_length - (phrases[p + offset_r].end - phrases[p - offset_l].start)

            if offset_l or offset_r:
                # Mark phrase span for merging
                spans_merge.append((p - offset_l, p + offset_r))
                print(phrase, phrase.root.dep_, doc[phrases[p - offset_l].start : phrases[p + offset_r].end])
                # Skip to end of phrase span
                skip_until += offset_r

    # Merge phrase spans
    phrases_merged, phrase_types_merged = merge_phrase_spans(phrases, phrase_types, spans_merge, doc)

    return phrases_merged, phrase_types_merged

def merge_nbor_of_singleton_phrase(nbor, singleton, offset, max_nbor_length):
    """
    Decide whether to merge neighbor of singleton (single-token) phrase.

    Evaluates conditions to determine if a neighboring phrase should be merged with a singleton phrase.

    Args:
        nbor (spacy.tokens.Span):
            Neighboring phrase.
        singleton (spacy.tokens.Span):
            Singleton phrase.
        offset (int):
            Absolute difference between indices of neighboring and singleton phrases.
        max_nbor_length (int):
            Maximum neighbor length for merging in terms of spaCy tokens.

    Returns:
        ret (bool):
            Whether to merge neighbor.
    """
    # Merge neighboring phrase if 0) not punctuation or space or too long AND
    # 1) neighbor is child of a preposition, OR
    # 2) neighbor is child of singleton AND a leaf phrase AND adjacent to singleton or a singleton itself, OR
    # 3) neighbor is conjunct and singleton phrase is coordinating conjunction
    ret = False
    # Check whether neighboring phrase is punctuation or space or is too long
    if any(is_not_punct_space(nbor)) and len(nbor) <= max_nbor_length:
        if nbor.root.head == singleton.root:
            # Neighbor is child of singleton phrase
            if singleton.root.dep_ == 'prep':
                # Neighbor is child of preposition, merge
                ret = True
            elif (offset == 1 or len(nbor) == 1) and nbor.n_lefts == 0 and nbor.n_rights == 0:
                # Neighbor is a leaf phrase and is either adjacent to singleton or a singleton itself
                ret = True
        elif singleton.root.dep_ == 'cc' and nbor.root.dep_ == 'conj':
            # Neighbor is conjunct and singleton is coordinating conjunction, merge
            ret = True

    return ret
