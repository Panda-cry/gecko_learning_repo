from flask import Flask
from flask_mail import Mail, Message
app = Flask(__name__)


#IAMP SETTINGS 
app.config['DEBUG'] = True
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = "petar.canic57@gmail.com"
app.config['MAIL_PASSWORD'] = 'drffwvsdxvkvbhoi'
app.config['MAIL_DEFAULT_SENDER'] = ("Petar Canic", "petar.canic57@gmail.com")
app.config['MAIL_MAX_EMAILS'] = None
app.config['MAIL_SUPRESS_SEND'] = False
app.config['MAIL_ASCII_ATTACHMENTS'] = False

#MAIL_DEFAULT_SENDER moze biti tuple da se doda name ko je sneder a moze i samo mail :D
# petarcanic79@gmail.com
# Flaskapp123#
#drffwvsdxvkvbhoi

mail = Mail(app)

#ili koristiti body ili html
#za vise mail da se salje u recipients samo dodajemo u listu i tjt
@app.route('/send_mail')
def send_mail():
    message = Message(
        subject="Hello form flask",
        recipients=["petar.canic57@gmail.com"],
        body="Today is friday and we are going to beach",
        html="<b> Danas je lep i suncan dan <b>"
    )
    with app.open_resource('dark_mountain.jpg') as picture:
        message.attach("dark_mountain.jpg", "image/jpg", picture.read())
        #naziv / kakav attachment se salje / content -> baytovi
            #  / mime type     /
    mail.send(message)
    return "Sended mail to me"



@app.route('/bulk')
def bulk():

    users = [{"name" : "pera" , "email": "email.email@gmail.com"}]

    with mail.connect() as connection:
        for user in users:
            message = Message("Bulk", recipients=[user.get('email')])
            message.body = f"Hello {user.get('name')}"
            connection.send(message)

#Bulk mail je ustvari kao sesija with mail.connection as conn:
#dok je to otvoreno kroz for petlju mozemo da iteriramo i da saljemo mailove ljudima
if __name__ == "__main__":
    app.run()