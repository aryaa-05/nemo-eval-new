"""Version of the answer-scoring rules. It is stamped on every run file and shown in every report header, so scores
made by different rules are never compared by accident. Bump it whenever a change can alter a verdict.

  unversioned  run files written before this stamp existed (scored with the first extractors)
  2            math answers: final-answer extraction (labels, bold values, '=' and is/are cues, no blind last-number
               fallback, ambiguity -> no_answer_found) and LaTeX normalisation (\\frac, \\sqrt, currency, thousands)
"""
SCORER_VERSION = "2"
UNVERSIONED = "unversioned"
