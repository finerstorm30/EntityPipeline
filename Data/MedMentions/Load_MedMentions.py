from bioc import biocxml
import gzip

def load_bioc(path):
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        collection = biocxml.load(fp)
    return collection.documents

def load_medmentions():
    TRAIN_FILE = "MedMentions/medmentions_st21pv_train.bioc.xml.gz"
    VAL_FILE = "MedMentions/medmentions_st21pv_val.bioc.xml.gz"
    TEST_FILE = "MedMentions/medmentions_st21pv_test.bioc.xml.gz"

    train_docs = load_bioc(TRAIN_FILE)
    val_docs = load_bioc(VAL_FILE)
    test_docs = load_bioc(TEST_FILE)

#   print(f"{len(train_docs)} training documents")
#   print(f"{len(val_docs)} validation documents")
#   print(f"{len(test_docs)} test documents")

    return train_docs, val_docs, test_docs
