from flask import Flask, jsonify, request, redirect, url_for
app = Flask(__name__)

app.config['DEBUG'] = True

# from database.models import Post

@app.route('/')
def index():
    return "You are on index page"


@app.route("/<name>")
def home(name):
    return f"<h1>Hello {name}<h1>"


# moguce je metodi da se da neka default vrednost
@app.route("/home", methods=["GET"], defaults={'number': 1})
@app.route("/home/<int:number>")
def number(number: int):
    return f'You gave me {number}'


@app.route("/json")
def json():
    return jsonify({'key': "data", 'key2': [1, 2, 3, 4, 5]})


@app.route('/query')
def query():
    name = request.args.get('name')
    return "hello your name is : {}".format(name)


@app.route('/jsonrequest', methods=["POST"])
def json_request():
    data = request.get_json()
    data['message'] = "I made it"
    return jsonify(data)


@app.route('/combined', methods=["GET", "POST"])
def combined():
    if request.method == "GET":
        return "We got the get request"
    else:
        return "We got the post request"


@app.route('/change_direction')
def change_direction():
    return redirect(url_for("combined"))

# pp = Post()
# pp.text = "SADASDASD"
#
# db.session.add(pp)
# db.session.commit()
if __name__ == "__main__":
    app.app_context().push()
    app.run(debug=True)

# nemoj da ti se url zove kao neka metoda !!!
# Pokretanje aplikacije moze flask run
# potrebno je export FLASK_APP= naziv modula koji pogrece app
# za debug moze FLASK_DEBUG=1
# a moze i if __name__ pa app.run(debug=True)
#konfiguracia app je po app.config
#moze se koristiti sesija kada se ubaci nesto u sesiion dict moze da se izbaci sve rute vide sesion i mogu da ga koriste
#ako se nesto slucajno izbaci bice error :D
#Moze debug kad izbaci error na web samo se stavi pin koji je iz terminala i do nekog dela se moze videti gde puca

# requests.form.get("nesto") kada iz forme zelimo da uzmemo nesto tj da submitujemo ili dodamo
# Svaka ruta po default je get moramo da dodamo u app.route methods=["GET","POST" ect]
