;;; Benjamin Kane 3-24-2023
;;; Functions for doing information retrieval using Python SentenceTransformer library

(in-package :information-retrieval)

(defvar +hf-embedding-prefix+ "https://api-inference.huggingface.co/pipeline/feature-extraction/")
(defvar +hf-similarity-prefix+ "https://api-inference.huggingface.co/models/")

(defparameter *api* nil)
(defparameter *model-name* "all-distilroberta-v1")
(defparameter *cross-encoder-name* "cross-encoder/ms-marco-electra-base")


(defun set-api (api)
;```````````````````````
; Used to set whether to use the HuggingFace API
;
  (setq *api* api)
) ; END set-api


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


(defun init (&key use-cross-encoder api-key)
;````````````````````````````````````````````
; Initializes by retrieving and storing the model for retrieval (and a cross-encoder model, if optionally specified).
; If *api* is set to t, then don't initialize local models, but set *model* to the appropriate model API URL.
;
  (cond
    ; Use API model
    (*api*
      (defparameter *model* *model-name*)
      (embedding-api api-key "init"))

    ; Use local models
    (t
      (py4cl:python-exec "import os")
      (py4cl:python-exec "os.environ['TOKENIZERS_PARALLELISM'] = 'false'")

      (if use-cross-encoder
        (py4cl:python-exec "from sentence_transformers import SentenceTransformer, CrossEncoder")
        (py4cl:python-exec "from sentence_transformers import SentenceTransformer"))

      (defparameter *model*
        (py4cl:python-eval (format nil "SentenceTransformer('~a')" *model-name*)))

      (if use-cross-encoder
        (defparameter *cross-encoder*
          (py4cl:python-eval (format nil "CrossEncoder('~a', max_length=512)" *cross-encoder-name*)))
        (defparameter *cross-encoder* nil))))

  t
) ; END init


(defun query-api (api-url api-key query)
;`````````````````````````````````````````
; Queries a HuggingFace API at a given URL, given an API key and a query (i.e., hash table)
;
  (let (data (header (make-hash-table :test #'equal)) response ret)
    (py4cl:python-exec "import json")
    (py4cl:python-exec "import requests")
    (setf (gethash "Authorization" header) (format nil "Bearer ~a" api-key))
    (setq data (py4cl:python-call "json.dumps" query))
    (setq response (py4cl:python-call "requests.request" "POST" api-url :headers header :data data))
    (setq ret (py4cl:python-method response "content.decode" "utf-8"))
    (setq ret (py4cl:python-call "json.loads" ret))
    ret
)) ; END query-api


(defun embedding-api (api-key documents)
;`````````````````````````````````````````
; Gets a list of embeddings for a list of documents, using the HuggingFace API.
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (let ((query (make-hash-table :test #'equal))
        (options (make-hash-table :test #'equal)))
    (setf (gethash "wait_for_model" options) t)
    (setf (gethash "inputs" query) documents)
    (setf (gethash "options" query) options)

    (query-api (concatenate 'string +hf-embedding-prefix+ *model*) api-key query)
)) ; END embedding-api


(defun similarity-api (api-key text documents)
;````````````````````````````````````````````````
; Gets a list of similarities between a given text and a list of documents, using the HuggingFace API.
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (let ((query (make-hash-table :test #'equal))
        (inputs (make-hash-table :test #'equal))
        (options (make-hash-table :test #'equal)))
    (setf (gethash "source_sentence" inputs) text)
    (setf (gethash "sentences" inputs) documents)
    (setf (gethash "wait_for_model" options) t)
    (setf (gethash "inputs" query) inputs)
    (setf (gethash "options" query) options)

    (query-api (concatenate 'string +hf-similarity-prefix+ *model*) api-key query)
)) ; END similarity-api


(defun embed-documents (documents &key filename append api-key indices)
;````````````````````````````````````````````````````````````````````````
; Embeds a list of documents (strings). If a filename is given, create a .CSV file containing the resulting
; embeddings for each document (or append to an existing .CSV file if :append t is given).
; If :indices (a list of strings of equal length as documents) is given, add as additional column to result.
;
; If using the API, do not embed documents, but still create a .CSV file containing the documents (if specified).
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (when (or (null indices) (not (listp indices)) (not (every #'stringp indices)) (not (= (length indices) (length documents))))
    (setq indices (loop for n from 0 below (length documents) collect (write-to-string n))))

  (when (null documents)
    (setq documents #())
    (setq indices #()))

  (let (embeddings data df)

    (cond
      ; Use API model
      (*api*
        (setq embeddings (embedding-api api-key documents)))

      ; Use local model
      (t
        (setq embeddings (py4cl:python-method *model* "encode" documents))))

    (when filename
      (py4cl:python-exec "import pandas as pd")
      (setq data (py4cl:python-call "list" (py4cl:python-call "zip" indices documents embeddings)))
      (setq df (py4cl:python-call "pd.DataFrame" data :columns #("indices" "document" "embedding")))
      (py4cl:python-call (py4cl:python-eval df ".to_csv") filename :index nil :mode (if append "a" "w") :header (if append nil t)))

    embeddings
)) ; END embed-documents


(defun load-data (documents+embeddings documents filename)
;````````````````````````````````````````````````````````````
; Loads data as dataframe from source. One of the following arguments must be provided:
;   documents+embeddings: a list of (<string> <embedding vector>) pairs.
;   documents: a list of documents (strings) to embed.
;   filename: the name of a CSV file containing 'document' and 'embedding' columns.
;
  (let (indices data df)
    (py4cl:python-exec "import pandas as pd")

    (cond
      (filename
        (setq df (py4cl:python-call "pd.read_csv" filename))
        (py4cl:python-exec df "['embedding']" "="
          (py4cl:python-eval df ".embedding.apply(eval)")))
      (documents+embeddings
        (setq indices (loop for n from 0 below (length documents+embeddings) collect (write-to-string n)))
        (setq data (py4cl:python-call "list" (py4cl:python-call "zip"
          indices
          (mapcar #'first documents+embeddings)
          (mapcar #'second documents+embeddings))))
        (setq df (py4cl:python-call "pd.DataFrame" data :columns #("indices" "document" "embedding"))))
      (documents
        (setq indices (loop for n from 0 below (length documents) collect (write-to-string n)))
        (setq data (py4cl:python-call "list" (py4cl:python-call "zip" documents (indices embed-documents documents))))
        (setq df (py4cl:python-call "pd.DataFrame" data :columns #("indices" "document" "embedding"))))
      (t (error "Must give one of :documents+embeddings, :documents or :filename as input")))

    df
)) ; END load-data


(defun retrieve (text &key (n 5) documents+embeddings documents filename api-key indices)
;`````````````````````````````````````````````````````````````````````````````````````````
; Given some text, retrieve the N most similar documents. One of the following 
; keyword arguments must be provided:
;   documents+embeddings: a list of (<string> <embedding vector>) pairs.
;   documents: a list of documents (strings) to embed.
;   filename: the name of a CSV file containing 'document' and 'embedding' columns.
;
; If :indices t is given, return the indices of the documents rather than the documents themselves.
;
  (when (or (not (boundp '*model*)) (null *model*))
    (init))

  (py4cl:python-exec "import numpy as np")
  (py4cl:python-exec "import pandas as pd")

  (py4cl:python-exec "def sim(x, np): return lambda y: np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))")

  (let (embedding df sim-func ret)
    ; Load data
    (setq df (load-data documents+embeddings documents filename))

    (when (py4cl:python-eval df ".empty")
      (return-from retrieve nil))

    ; Get text embedding 
    (cond
      ; Use API model
      (*api*
        (setq embedding (embedding-api api-key text)))

      ; Use local model
      (t
        (setq embedding (py4cl:python-method *model* "encode" text))))

    ; Get similarity
    (setq sim-func (py4cl:python-call "sim" embedding (py4cl:python-eval "np")))
    (py4cl:python-exec df "['similarity']" "="
      (py4cl:python-call (py4cl:python-eval df ".embedding.apply") sim-func))

    ; Sort by similarity and return top n
    (setq ret (py4cl:python-call (py4cl:python-eval df ".sort_values") "similarity" :ascending 0))
    (setq ret (py4cl:python-method ret "head" n))

    (if indices
      (py4cl:python-call "list" (py4cl:python-eval ret ".indices"))
      (py4cl:python-call "list" (py4cl:python-eval ret ".document")))
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