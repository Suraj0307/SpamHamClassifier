from flask import Flask, request, render_template
from test import transform_text, vectorize_word, predict

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':

        text = request.form.get('message')

        transformed_text = transform_text(text)
        vector = vectorize_word(transformed_text)
        results = predict(vector)

        if results == 1:
            results = "Yes it is a Spam Message"
        else:
            results = 'No it is not a spam message'
        return render_template('index.html', prediction=results)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()

