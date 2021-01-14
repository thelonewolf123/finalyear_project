from flask import Flask, render_template, request, redirect, abort
from flask_sqlalchemy import SQLAlchemy

from datetime import datetime


app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://fxvhedps:J5ZQxW0jMmthfk5s5LioLYiDMuyqAZCM@arjuna.db.elephantsql.com:5432/fxvhedps'

db = SQLAlchemy(app)


class Person(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    task = db.Column(db.String(25000))
    date = db.Column(db.DateTime)


@app.route('/',methods=['GET'])
def index():

    todos = Todo.query.all()

    return render_template('index.html',title='Flask App',todos=todos)


if __name__ == '__main__':
    app.run(debug=True)
