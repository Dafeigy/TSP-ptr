from flask import Flask,render_template
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/get_sample', methods=['GET', 'POST'])
def get_sample():
    pass

@app.route('/get_result', methods=['GET', 'POST'])
def get_result():
    pass

if __name__ == '__main__':
    app.run(debug=True)