import os, datetime, csv, time, subprocess
import pandas as pd
from nltk import tokenize
from bert_serving.client import BertClient

subprocess.Popen(["bert-serving-start","-model_dir","/src/model/", "-num_worker=1", "-max_seq_len=15"])

def load_from_csv(file):
    df = pd.read_csv(file, error_bad_lines=False, nrows=1000)
    df = df.rename(str.lower, axis='columns')
    df['title'] = df['title'].apply(lambda x: x.replace("'s", " " "s").replace("\n"," "))
    df['title'] = df['title'].apply(lambda x: tokenize.sent_tokenize(x))
    bc = BertClient()
    df['encoded'] = df['title'].apply(lambda x: bc.encode(x))
    return df

def create_tsv(embeddings, filepath, as_array):
    with open(filepath, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for e in embeddings:
            e = [e] if as_array else e
            tsv_writer.writerow(e)

def run_calculations(input_file):
    df=load_from_csv(input_file)
    # Creates a matrix of embeddings of above df file
    embeddings = [i[0] for i in df['encoded']][1:]
    folder_name = "/src/results/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(folder_name)
    create_tsv(embeddings, folder_name + "/tensor.tsv", False)
    # Creates a labels file from above df
    df['short_str'] = df['title'].str.slice(0,10)
    metadata=df['short_str'].iloc[1:]
    #Produces a tsv file from above
    create_tsv(metadata, folder_name + "/metadata.tsv", True)

############################## BEGIN FLASK ##############################
from flask import Flask, render_template, request, send_file
from multiprocessing import Process
app = Flask(__name__)

@app.route('/')
def upload():
    return render_template("upload.html")

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save("/src/upload/" + f.filename)
        heavy_process = Process(
            target=run_calculations,
            args=("/src/upload/" + f.filename,),
            daemon=True
        )
        heavy_process.start()
        return render_template("success.html", name=f.filename)

@app.route('/overview', methods=['GET'])
def overview():
    data = os.walk("/src/results")
    return render_template("overview.html", data=data)

@app.route('/download/<path>', methods=['GET'])
def download(path):
    file_path = path.replace("?", "/")
    test = "/src/results/metadata.tsv"
    return send_file(file_path,
                    mimetype='text/tsv',
                    attachment_filename=file_path.split("/")[-1],
                    as_attachment=True)

if __name__ == "__main__":
  app.run(host="0.0.0.0")
############################## END FLASK ##############################
