from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from numpy import array, argmax, expand_dims
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from os import listdir
from tensorflow.keras.models import load_model
from cv2 import resize, cvtColor, COLOR_GRAY2BGRA, COLOR_BGRA2BGR
from PIL import Image

#importing the model
new_model = load_model('pokemon_classifier_dropout=8,6_lr=0.0001.h5')

#making a dictionary with the keys as labels
label_dict = {'Abra': 0,
 'Aerodactyl': 1,
 'Alakazam': 2,
 'Alolan Sandslash': 3,
 'Arbok': 4,
 'Arcanine': 5,
 'Articuno': 6,
 'Beedrill': 7,
 'Bellsprout': 8,
 'Blastoise': 9,
 'Bulbasaur': 10,
 'Butterfree': 11,
 'Caterpie': 12,
 'Chansey': 13,
 'Charizard': 14,
 'Charmander': 15,
 'Charmeleon': 16,
 'Clefable': 17,
 'Clefairy': 18,
 'Cloyster': 19,
 'Cubone': 20,
 'Dewgong': 21,
 'Diglett': 22,
 'Ditto': 23,
 'Dodrio': 24,
 'Doduo': 25,
 'Dragonair': 26,
 'Dragonite': 27,
 'Dratini': 28,
 'Drowzee': 29,
 'Dugtrio': 30,
 'Eevee': 31,
 'Ekans': 32,
 'Electabuzz': 33,
 'Electrode': 34,
 'Exeggcute': 35,
 'Exeggutor': 36,
 'Farfetchd': 37,
 'Fearow': 38,
 'Flareon': 39,
 'Gastly': 40,
 'Gengar': 41,
 'Geodude': 42,
 'Gloom': 43,
 'Golbat': 44,
 'Goldeen': 45,
 'Golduck': 46,
 'Golem': 47,
 'Graveler': 48,
 'Grimer': 49,
 'Growlithe': 50,
 'Gyarados': 51,
 'Haunter': 52,
 'Hitmonchan': 53,
 'Hitmonlee': 54,
 'Horsea': 55,
 'Hypno': 56,
 'Ivysaur': 57,
 'Jigglypuff': 58,
 'Jolteon': 59,
 'Jynx': 60,
 'Kabuto': 61,
 'Kabutops': 62,
 'Kadabra': 63,
 'Kakuna': 64,
 'Kangaskhan': 65,
 'Kingler': 66,
 'Koffing': 67,
 'Krabby': 68,
 'Lapras': 69,
 'Lickitung': 70,
 'Machamp': 71,
 'Machoke': 72,
 'Machop': 73,
 'Magikarp': 74,
 'Magmar': 75,
 'Magnemite': 76,
 'Magneton': 77,
 'Mankey': 78,
 'Marowak': 79,
 'Meowth': 80,
 'Metapod': 81,
 'Mew': 82,
 'Mewtwo': 83,
 'Moltres': 84,
 'MrMime': 85,
 'Muk': 86,
 'Nidoking': 87,
 'Nidoqueen': 88,
 'Nidorina': 89,
 'Nidorino': 90,
 'Ninetales': 91,
 'Oddish': 92,
 'Omanyte': 93,
 'Omastar': 94,
 'Onix': 95,
 'Paras': 96,
 'Parasect': 97,
 'Persian': 98,
 'Pidgeot': 99,
 'Pidgeotto': 100,
 'Pidgey': 101,
 'Pikachu': 102,
 'Pinsir': 103,
 'Poliwag': 104,
 'Poliwhirl': 105,
 'Poliwrath': 106,
 'Ponyta': 107,
 'Porygon': 108,
 'Primeape': 109,
 'Psyduck': 110,
 'Raichu': 111,
 'Rapidash': 112,
 'Raticate': 113,
 'Rattata': 114,
 'Rhydon': 115,
 'Rhyhorn': 116,
 'Sandshrew': 117,
 'Sandslash': 118,
 'Scyther': 119,
 'Seadra': 120,
 'Seaking': 121,
 'Seel': 122,
 'Shellder': 123,
 'Slowbro': 124,
 'Slowpoke': 125,
 'Snorlax': 126,
 'Spearow': 127,
 'Squirtle': 128,
 'Starmie': 129,
 'Staryu': 130,
 'Tangela': 131,
 'Tauros': 132,
 'Tentacool': 133,
 'Tentacruel': 134,
 'Vaporeon': 135,
 'Venomoth': 136,
 'Venonat': 137,
 'Venusaur': 138,
 'Victreebel': 139,
 'Vileplume': 140,
 'Voltorb': 141,
 'Vulpix': 142,
 'Wartortle': 143,
 'Weedle': 144,
 'Weepinbell': 145,
 'Weezing': 146,
 'Wigglytuff': 147,
 'Zapdos': 148,
 'Zubat': 149
 }




app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')



@app.route('/uploadPokemon')
def uploadPokemon():
    return render_template('uploadPokemon.html')



@app.route('/classifiedPokemon', methods = ['GET', 'POST'])
def classifiedPokemon():
    if request.method == 'POST':
       data = request.files['file']
       img = Image.open(request.files['file'])
       img = array(img)
       img = resize(img,(224,224))

       #coverting channel 1 image to channel 3
       if len(img.shape)==2:
           img = cvtColor(img, COLOR_GRAY2BGRA)

       #converting channel 4 image to channel 3
       if len(img.shape) > 2 and img.shape[2] == 4:
           #convert the image from RGBA2RGB
           img = cvtColor(img, COLOR_BGRA2BGR)

       preprocessed_image = prepare_image(img)
       predictions = new_model.predict(preprocessed_image)
       out = argmax(predictions)
       pokemon = list(label_dict.keys())[list(label_dict.values()).index(out)]

       #getting pokemon image directory
       list_dir_string = f'static/{pokemon}'
       pokemon_image_directory = listdir(list_dir_string)

       #returning image
       pic1_name = pokemon_image_directory[0]
       pic1_location = f'static/{pokemon}/{pic1_name}'


       return render_template('classifiedPokemon.html', pokemon = pokemon, pic1_location=pic1_location)



@app.route('/pokemonList')
def pokemonList():
    return render_template('pokemonList.html')


@app.route('/uploadOwn')
def uploadOwn():
    return render_template('uploadOwn.html')



@app.route('/classifiedOwn', methods = ['GET', 'POST'])
def classifiedOwn():
    if request.method == 'POST':
       data = request.files['file']
       img = Image.open(request.files['file'])
       img = array(img)
       img = resize(img,(224,224))

       #converting channel 1 image to channel 3
       if len(img.shape)==2:
           img = cvtColor(img, COLOR_GRAY2BGRA)

       #converting channel 4 image to channel 3
       if len(img.shape) > 2 and img.shape[2] == 4:
           #convert the image from RGBA2RGB
           img = cvtColor(img, COLOR_BGRA2BGR)


       preprocessed_image = prepare_image(img)
       predictions = new_model.predict(preprocessed_image)
       out = argmax(predictions)
       pokemon = list(label_dict.keys())[list(label_dict.values()).index(out)]

       #getting pokemon image directory
       list_dir_string = f'static/{pokemon}'
       pokemon_image_directory = listdir(list_dir_string)

       #returning image
       pic1_name = pokemon_image_directory[0]
       pic1_location = f'static/{pokemon}/{pic1_name}'

       return render_template('classifiedOwn.html', pokemon = pokemon, pic1_location=pic1_location)


def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded_dims = expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)



if __name__ == '__main__':
   app.run()
