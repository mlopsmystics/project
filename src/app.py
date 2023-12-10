from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder='templates')

with open('data.json') as f:
    data = f.read()

@app.route("/")
def index():
    return render_template('index.html', title="page", jsonfile=jsonify(data))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
