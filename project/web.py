from flask import Flask, render_template, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import search
from preprocessing import load_data
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'labread'
form = ''
result = []
req = ''
data = load_data()


class RequestForm(FlaskForm):
    request = StringField('', validators=[DataRequired()])
    submit = SubmitField('инфопоискать!')


class PageForm(FlaskForm):
    request_p = StringField('', validators=[DataRequired()])
    submit_p = SubmitField('к странице')


@app.route('/')
def hello():
    return redirect("/bm-25")


@app.route('/<method>', methods=['GET', 'POST'])
def index(method):
    global form, result, req
    form = RequestForm()
    if form.validate_on_submit():
        req = form.request.data
        result = search.search(req, data, method=method)
        form.request.data = ''
        return render_template('result.html', req=req, result=result)
    else:
        req = None
        result = []
        return render_template('main.html', req=req, form=form, result=result, method=method)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=False)
