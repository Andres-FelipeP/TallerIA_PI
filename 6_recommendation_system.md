## Sistema de recomendación

Para el sistema de recomendación de películas se utilizará la API de generación de embeddings de openAI. De forma general, un embedding es una representación numérica de cualquier fenómeno, puede ser una  imagen, un video, o como en este caso, texto.

![Fork 1](imgs/sr1.svg)

__Concepto Básico:__ Un embedding es básicamente una proyección de un objeto no vectorial en un espacio vectorial. En el contexto de PLN (procesamiento de lenguaje natural), los embeddings convierten palabras o frases en vectores de números reales.

__Utilidad:__ Estos vectores capturan la semántica y las relaciones contextuales entre las palabras. Palabras con significados similares o que a menudo aparecen en el mismo contexto tendrán embeddings similares, es decir, sus vectores estarán cerca en el espacio vectorial.

Se utilizará una medida de similitud muy conocida para calcular qué tan parecido es un prompt de entrada con las descripciones de las películas. 

La similitud de coseno mide el coseno del ángulo entre dos vectores, en este caso, dos embeddings. Es una métrica popular para calcular la similitud en espacios de alta dimensión como el de los embeddings.

Funciona de la siguiente manera:

__Fórmula:__ La similitud de coseno entre dos vectores AA y BB se calcula como el producto punto de los vectores dividido por el producto de sus magnitudes:

$similitud(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$

Donde $A⋅B$ es el producto punto de los vectores y $∥A∥$ y $∥B∥$ son las magnitudes (o normas) de los vectores A y B respectivamente.

__Rango de Valores:__ La similitud de coseno produce un valor entre -1 y 1. Un valor de 1 indica que los vectores son idénticos en dirección, un valor de -1 indica que son opuestos y un valor de 0 indica que son ortogonales (no relacionados).

__Aplicación a Embeddings:__ Cuando se utiliza con embeddings, la similitud de coseno permite determinar qué tan similares son semánticamente dos palabras o frases. Si sus embeddings son cercanos en el espacio vectorial (es decir, tienen un ángulo pequeño entre ellos), su similitud de coseno será cercana a 1.

En resumen, la similitud de coseno compara la orientación de dos vectores en lugar de su magnitud, siendo una métrica esencial para evaluar la similitud semántica entre embeddings en el PLN.

El script [movie_similarities.py](movie_similarities.py) muestra cómo podemos utilizar los embeddings y la similitud de coseno para encontrar similitudes entre películas.

Al ejecutar el Script puede ver lo siguiente:

![Fork 1](imgs/rs2a.png)

En este caso, los embeddings nos dicen que Salvar al soldado Ryan es más parecido a la Lista Schindler. Tiene sentido porque las dos películas son de la segunda guerra mundial. 

En las líneas 34 a 42 del script se generan los embeddings de las descripciones de las películas en la base de datos y se almacenan en el archivo [movie_descriptions_embeddings.json](movie_descriptions_embeddings.json).

El script [movie_recommendations.py](movie_recommendations.py) muestra cómo podemos utilizar los embeddings y la similitud de coseno para recomendar películas a partir de un prompt.

Al ejecutar el Script donde el prompt dice ``película de un pianista`` puede ver lo siguiente:

![Fork 1](imgs/sr3a.png)

En este caso, los embeddings nos dicen que si queremos ver una película de un pianista deberíamos ver El Pianista. Intente com prompts diferentes para encontrar películas menos obvias. 

Ahora, se deben agregar estos embeddings a la base de datos para poder hacer este tipo de búsquedas. Dado que un vector no se puede agregar directamente a la base de datos, deberá crear un archivo binario que se pueda almacenar como un campo de tipo __BinaryField__.

En el siguiente ejemplo puede ver cómo crear un archivo binario a partir de una lista y cómo recuperar la lista a partir del archivo binario, este ejemplo es ilustrativo, no es necesario ejecutarlo.

````python
import numpy as np
from openai import OpenAI

_ = load_dotenv('api_keys.env')
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('openai_api_key'),
)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#Generar binario
desc = "película de la segunda guerra mundial"
emb = get_embedding(desc)

#Recuperar lista a partir del archivo binario
emb_binary = np.array(emb).tobytes()
rec_emb = list(np.frombuffer(emb_binary, dtype=arr.dtype))
````

Modifique el modelo Movie de la siguiente forma:

````python
from django.db import models
import numpy as np

# create your models here
def get_default_array():
  default_arr = np.random.rand(1536)  # Adjust this to your desired default array
  return default_arr.tobytes()


class Movie(models.Model): 
    title = models.CharField(max_length=100)
    description = models.CharField(max_length=250) 
    image = models.ImageField(upload_to='movie/images/', default = 'movie/images/default.jpg') 
    url = models.URLField(blank=True)
    genre = models.CharField(blank=True, max_length=250)
    year = models.IntegerField(blank=True, null=True)
    emb = models.BinaryField(default=get_default_array())

    def __str__(self): 
        return self.title

````

Note que está agregando un campo ``emb`` de tipo ``models.BinaryField``

Recuerde que cada que se hace una modificación al modelo se deben hacer las migraciones.

Finalmente, para modificar los items de la base de datos (en este caso agregar los embeddings), debe crear un archivo ``add_embeddings_db.py`` en la carpeta ``movie/management/command`` y ejecutarlo. 

````python
from django.core.management.base import BaseCommand
from movie.models import Movie
import json
import os
import numpy as np

class Command(BaseCommand):
    help = 'Modify path of images'

    def handle(self, *args, **kwargs):
        ##Código para leer los embeddings del archivo movie_descriptions_embeddings.json
        json_file_path = '../movie_descriptions_embeddings.json'
        # Load data from the JSON file
        with open(json_file_path, 'r') as file:
            movies = json.load(file)       
  
        for movie in movies:
            emb = movie['embedding']
            emb_binary = np.array(emb).tobytes()
            item = Movie.objects.filter(title = movie['title']).first()
            item.emb = emb_binary
            item.save()
        
        self.stdout.write(self.style.SUCCESS(f'Successfully updated item embeddings'))        
        
````

Para comprobar que los embeddings se crearon correctamente, debe crear un archivo ``check_embeddings_db.py`` en la ruta ``movie/management/command``. El archivo lo puede ver en [check_embeddings_db.py](DjangoProjectBase/movie/management/commands/check_embeddings_db.py).
Cuando lo ejecute deberá ver en la consola algo de esta forma:

![Fork 1](imgs/sr3.png)

Finalmente, para comprobar que el sistema de recomendación funciona dentro de la aplicación, cree un archivo ``check_rec_sys.py`` en la ruta ``movie/management/command``. El archivo lo puede ver en [check_rec_sys_db.py](DjangoProjectBase/movie/management/commands/check_rec_sys.py).

Cuando lo ejecute deberá ver en la consola algo de esta forma:

![Fork 1](imgs/sr4a.png)

__Nota:__ El archivo [movie_descriptions_gemini.py](movie_descriptions_gemini.py) muestra como generar embeddings y calcular similitudes utilizando la api de gemini. Este paso es opcional pero puede ser útil para el desarrollo de sus proyectos.
