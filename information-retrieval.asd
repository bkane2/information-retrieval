;; Information Retrieval
;; Packaged on 2023-3-24

(asdf:defsystem :information-retrieval
  :name "information-retrieval"
  :version "0.0.1"
  :author "Benjamin Kane"
  :depends-on (:py4cl)
  :components ((:file "package")
               (:file "information-retrieval")))