;;; Benjamin Kane 3-24-2023
;;; Functions for doing information retrieval using Python SentenceTransformer library

(in-package :information-retrieval)

(defparameter *model-name* "all-distilroberta-v1")
(defparameter *cross-encoder-name* "cross-encoder/ms-marco-electra-base")


(defun set-model (model-name)
;```````````````````````````````
; Used to set a different retrieval model from the default.
;
  (setq *model-name* model-name)
) ; END set-model


(defun set-cross-encoder (cross-encoder-name)
;```````````````````````````````````````````````
; Used to set a different cross-encoder model from the default.
;
  (setq *cross-encoder-name* cross-encoder-name)
) ; END set-cross-encoder


(defun init (&key use-cross-encoder)
;`````````````````````````````````````
; Initializes by retrieving and storing the model for retrieval (and a cross-encoder model, if optionally specified).
;
  (if use-cross-encoder
    (py4cl:python-exec "from sentence_transformers import SentenceTransformer, CrossEncoder")
    (py4cl:python-exec "from sentence_transformers import SentenceTransformer"))

  (defparameter *model*
    (py4cl:python-eval (format nil "SentenceTransformer('~a')" *model-name*)))

  (if use-cross-encoder
    (defparameter *cross-encoder*
      (py4cl:python-eval (format nil "CrossEncoder('~a', max_length=512)" *cross-encoder-name*)))
    (defparameter *cross-encoder* nil))

  t
) ; END init


(defun embed-documents (documents &key filename append)
;`````````````````````````````````````````````````````````
; Embeds a list of documents (strings). If a filename is given, create a .CSV file containing the resulting
; embeddings for each document (or append to an existing .CSV file if :append t is given).
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (let (embeddings data df)
    (setq embeddings (py4cl:python-method *model* "encode" documents))
    
    (when filename
      (py4cl:python-exec "import pandas as pd")
      (setq data (py4cl:python-call "list" (py4cl:python-call "zip" documents embeddings)))
      (setq df (py4cl:python-call "pd.DataFrame" data :columns #("document" "embedding")))
      (py4cl:python-call (py4cl:python-eval df ".to_csv") filename :index nil :mode (if append "a" "w") :header (if append nil t)))
  
    embeddings
)) ; END embed-documents


(defun retrieve (text &key (n 5) documents+embeddings documents filename)
;```````````````````````````````````````````````````````````````````````````
; Given some text, retrieve the N most similar documents. One of the following 
; keyword arguments must be provided:
;   documents+embeddings: a list of (<string> <embedding vector>) pairs.
;   documents: a list of documents (strings) to embed.
;   filename: the name of a CSV file containing 'document' and 'embedding' columns.
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (py4cl:python-exec "import numpy as np")
  (py4cl:python-exec "import pandas as pd")

  (py4cl:python-exec "def sim(x, np): return lambda y: np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))")

  (let (embedding data df sim-func ret)
    (cond
      (filename
        (setq df (py4cl:python-call "pd.read_csv" filename))
        (py4cl:python-exec df "['embedding']" "="
          (py4cl:python-eval df ".embedding.apply(eval)")))
      (documents+embeddings
        (setq data (py4cl:python-call "list" (py4cl:python-call "zip"
          (mapcar #'first documents+embeddings)
          (mapcar #'second documents+embeddings))))
        (setq df (py4cl:python-call "pd.DataFrame" data :columns #("document" "embedding"))))
      (documents
        (setq data (py4cl:python-call "list" (py4cl:python-call "zip" documents (embed-documents documents))))
        (setq df (py4cl:python-call "pd.DataFrame" data :columns #("document" "embedding"))))
      (t (error "Must give one of :documents+embeddings, :documents or :filename as input")))

    (setq embedding (py4cl:python-method *model* "encode" text))
    (setq sim-func (py4cl:python-call "sim" embedding (py4cl:python-eval "np")))

    (py4cl:python-exec df "['similarity']" "="
      (py4cl:python-call (py4cl:python-eval df ".embedding.apply") sim-func))

    (py4cl:python-exec df "['similarity']" "="
          (py4cl:python-call (py4cl:python-eval df ".embedding.apply") sim-func))

    (setq ret (py4cl:python-call (py4cl:python-eval df ".sort_values") "similarity" :ascending 0))
    (setq ret (py4cl:python-method ret "head" n))

    (py4cl:python-call "list" (py4cl:python-eval ret ".document"))
)) ; END retrieve


(defun rerank (text documents &key (n 1))
;```````````````````````````````````````````
; Reranks a set of documents given a text using a cross-encoder model.
;
  (when (or (not (boundp '*cross-encoder*)) (null *cross-encoder*))
    (init :use-cross-encoder t))

  (py4cl:python-exec "import pandas as pd")

  (py4cl:python-exec "def score(x, model): return lambda y: model.predict((x, y))")

  (let (df score-func ret)
    (setq df (py4cl:python-call "pd.DataFrame" documents :columns #("document")))

    (setq score-func (py4cl:python-call "score" text *cross-encoder*))

    (py4cl:python-exec df "['score']" "="
      (py4cl:python-call (py4cl:python-eval df ".document.apply") score-func))

    (setq ret (py4cl:python-call (py4cl:python-eval df ".sort_values") "score" :ascending 0))
    (setq ret (py4cl:python-method ret "head" n))

    (py4cl:python-call "list" (py4cl:python-eval ret ".document"))
)) ; END rerank


(defun retrieve+rerank (text &key (n-candidate 5) (n-result 1) documents+embeddings documents filename)
;````````````````````````````````````````````````````````````````````````````````````````````````````````
; Retrieves a list of the n-candidate most similar sentences using the sentence transformer model, and then
; chooses the top n-result after re-ranking using the cross-encoder model. One of the following  keyword
; arguments must be provided:
;   documents+embeddings: a list of (<string> <embedding vector>) pairs.
;   documents: a list of documents (strings) to embed.
;   filename: the name of a CSV file containing 'document' and 'embedding' columns.
;
  (let (candidates ret)
    (setq candidates (retrieve text :n n-candidate
                        :documents+embeddings documents+embeddings
                        :documents documents
                        :filename filename))
    (setq ret (rerank text candidates :n n-result))
)) ; END retrieve+rerank