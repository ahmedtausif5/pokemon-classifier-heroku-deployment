# app.py — minimal changes for Streamlit compatibility (no Flask)
from numpy import array, argmax, expand_dims, argsort
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from os import listdir, path
from tensorflow.keras.models import load_model
from cv2 import resize, cvtColor, COLOR_GRAY2BGRA, COLOR_BGRA2BGR
from PIL import Image
import streamlit as st

# importing the model (unchanged)
new_model = load_model('pokemon_classifier_dropout=8,6_lr=0.0001.h5')

# making a dictionary with the keys as labels (unchanged)
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
              'Zubat': 149}

def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded_dims = expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

# ---------------------------
# Streamlit UI (replaces Flask routes/templates)
# ---------------------------
st.set_page_config(page_title="Pokémon Identifier", page_icon="🧪")
st.title("🧪 Pokémon Identifier")

mode = st.radio(
    "Choose mode",
    ["Classify (single result, 0.70 threshold)", "Classify (top-3)"],
    index=0
)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
if uploaded is not None:
    try:
        img = Image.open(uploaded)
    except Exception:
        img = None
        st.error("Could not read the uploaded file as an image.")

    if img is not None:
        st.image(img, caption="Uploaded image", use_column_width=True)

        # --- original preprocessing steps preserved ---
        img_np = array(img)
        img_np = resize(img_np, (224, 224))

        # converting channel 1 image to channel 3
        if len(img_np.shape) == 2:
            img_np = cvtColor(img_np, COLOR_GRAY2BGRA)

        # converting channel 4 image to channel 3
        if len(img_np.shape) > 2 and img_np.shape[2] == 4:
            # convert the image from RGBA2RGB
            img_np = cvtColor(img_np, COLOR_BGRA2BGR)

        preprocessed_image = prepare_image(img_np)
        predictions = new_model.predict(preprocessed_image)
        out = argmax(predictions)
        pokemon = list(label_dict.keys())[list(label_dict.values()).index(out)]

        if mode == "Classify (single result, 0.70 threshold)":
            prob_1 = predictions[0][out]
            if prob_1 >= 0.70:
                st.subheader(f"Prediction: {pokemon} ({prob_1*100:.2f}%)")

                # Show a sample image from static/<pokemon>/ if available
                static_dir = f'static/{pokemon}'
                if path.isdir(static_dir):
                    files = listdir(static_dir)
                    if files:
                        st.image(f"{static_dir}/{files[0]}", caption=f"Sample: {pokemon}")
                else:
                    st.info("No static sample image found.")
            else:
                st.warning("Could not classify with ≥ 70% confidence.")

        else:  # "Classify (top-3)"
            order_indexes = (argsort(predictions))
            top_3_indexes = [order_indexes[0][-1], order_indexes[0][-2], order_indexes[0][-3]]

            prob_1 = round(predictions[0][top_3_indexes[0]] * 100, 2)
            prob_2 = round(predictions[0][top_3_indexes[1]] * 100, 2)
            prob_3 = round(predictions[0][top_3_indexes[2]] * 100, 2)

            top_3_prob_list = [prob_1, prob_2, prob_3]

            scale_1 = (prob_1 / prob_1) if prob_1 else 0.0
            scale_2 = (prob_2 / prob_1) if prob_1 else 0.0
            scale_3 = (prob_3 / prob_1) if prob_1 else 0.0

            scaled_list = [scale_1, scale_2, scale_3]

            pokemon_names = []
            for idx in top_3_indexes:
                pokemon_names.append(list(label_dict.keys())[list(label_dict.values()).index(idx)])

            st.subheader("Top-3")
            for i, (name, p) in enumerate(zip(pokemon_names, top_3_prob_list), start=1):
                st.write(f"**{i}. {name}** — {p:.2f}%")
                st.progress(min(max(scaled_list[i-1], 0.0), 1.0))

            # Show sample images from static/<pokemon> if available
            cols = st.columns(3)
            for col, name in zip(cols, pokemon_names):
                static_dir = f"static/{name}"
                with col:
                    if path.isdir(static_dir):
                        files = listdir(static_dir)
                        if files:
                            st.image(f"{static_dir}/{files[0]}", caption=name)
                    else:
                        st.info(f"No sample for {name}")
else:
    st.info("👆 Upload an image to classify.")

# No __main__ block or Flask app.run() — Streamlit runs this script directly.
