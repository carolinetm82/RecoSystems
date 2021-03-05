from flask import Flask, render_template,request
import pandas as pd
import numpy as np
from functions import *

app = Flask(__name__)

# Create the first page
@app.route('/')
def index2():
    plays = pd.read_csv('/home/caroline09/projects/Recommendations/user_artists.dat',sep='\t')
    artists = pd.read_csv('/home/caroline09/projects/Recommendations/artists.dat',sep='\t',usecols=['id','name'])

    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    # List of the 50 first artists
    artists_tuple=list(sorted(ap['name'].unique()[:50]))

    return render_template('index.html',your_list=artists_tuple)

# Create the page of results
@app.route('/results', methods=['POST'])
def results():
    plays = pd.read_csv('/home/caroline09/projects/Recommendations/user_artists.dat',sep='\t')
    artists = pd.read_csv('/home/caroline09/projects/Recommendations/artists.dat',sep='\t',usecols=['id','name'])

    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    artists_tuple=list(sorted(ap['name'].unique()[:50]))
    artists_tuple_glo=artists_tuple

    var_1 = request.form.getlist("selectedArtist")
    print(var_1)
    var_2=recherche(var_1,artists_tuple_glo)
    var_3=recommender(var_2)
    # Get the results of the artist search
    return render_template('results.html', art_selected1=var_3)



if __name__ == '__main__':
   app.run(debug = True)