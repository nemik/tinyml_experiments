import tensorshow

rec = "dataset/val.record-00000-of-00010"
out = rec+".html"
# The column labels of `df` are the features of the tf.train.example protobufs.
df = tensorshow.html_file_from(rec, out, limit=100)
