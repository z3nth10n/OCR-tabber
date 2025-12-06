Me gustaría que primeramente con opencv reconociera ciertas partes de la imagen (examples/sample-tab.png), para ello identificaremos diferentes partes de la tablatura que más abajo te describiré.

Para ello vamos a descomponer en dos partes el problema, primeramente usar opencv para identificar diferentes partes de la tablatura y luego usar tesseract OCR para identificar los números de los trastes, la afinación, el tiempo, los bpm dentro de la tablatura, ya que estos son textos.

Además, me gustaría que generes un segundo script de depuración que dibuje rectángulos de diferentes colores para ver si el script de opencv es capaz de identificar correctamente todas las partes.

Dentro de la tablatura de la imagen que te adjunto (examples/sample-tab.png), hay que identficar varias partes de la tablatura para que el programa pueda funcionar correcta y eficientemente.

A continuació, te explicaré cada parte y su imagen de referencia, recortada directamente de examples/sample-tab.png (a ecepción de empty_tab.png)

* (1) Lo primero sería definir donde están las esquinas del la tablatura (que como observarás en la imagen la tablatura tiene una forma rectangular y está dividida en 6 lineas horizontales que corresponden a las 6 cuerdas de la guitarra), imagen de referencia: prompts/images/empty_tab.png
* tenemos linas verticales, que son las que dividen cada compás. Imagen de referencia: prompts/images/compass_divider.png
* (2) A su izquierda tendrás letras que corresponderán a la afinación que es usada en dicha tablatura, para este caso tenemos E estándar. Imagen de referencia: prompts/images/tuning.png
* (3) Luego identificaremos cada uno de los compases, que estos estarán divididos por líneas verticales que normalmente suelen ser equidistantes. Imagen de referencia: prompts/images/compass_divider.png
* (4) Por otro lado, también tendremos que identificar por cada una de las lineas los números que van apareciendo y que luego haya una cohesión entre todas las cuerdas, hay que respetar el orden en el que van apareciendo cada uno para que así podamos diferenciar bien el tiempo en el que se van sucediendo cada uno de estos números, ya que al final el eje de la X en una tablatura representa tiempo. Para este cso, la imagen de referencia es sample-tab, de ahí debes analizar los números en negrita que van apareciendo, ya que estos son las notas de la tablatura.
* (5) Por otro lado, también tenemos que diferenciar diferentes figuras, por un lado tenemos 4 y debajo otro 4, que representa que el compás es de 4 por 4. Imagen de referencia: prompts/images/time.png
* (6) Debajo de la tablatura tenemos líneas verticales y horizontales que van separadas cada compás (lineas verticales que hemos explicado en el paso 2 de esta lista). Esto representa que son corcheas y por eso por cada compás aparecen 4 y 4, para hacer los 8 tiempos de un compás de 4 por cuatro, aparecerán otras figuras pero por el momento identifica está después veremos otras que puedan aparecer 
* (7) Otros textos pueden ser los de PM que significa Palm mute  
* (8) También aparece una corchea con un 180 diciendo el BPM de la tablatura

Si quieres sirvete del preprocesamiento que ya hay si te sirve de algo, y si no, directamente descartalo y usa tu propio enfoque.

Habiendo identificado correctamente cada figura el siguiente paso será usar el txt (examples/validation.txt) donde te explicré a continuación el resultado el equivalente en texto al que quiero que conviertas dicha imagen. 

Procedo a epxlicarte como quiero que generes el txt.

Como se puede ver

Tenemos metadatos de la información de la tablatura:

Song: OCR Validation
Artist: Visual Tab
BPM: 180
Time: 4/4

Por el momento song y artist rellenalos con informacion arbitraria puesto que en la magen (examples/sample-tab.png) no se da dicha información.

Tenemos por un lado la afinación:

e|
B|
G|
D|
A|
E|

Por otro lado tenemos los compases:

| 1                         | 2                         | 3                         | 4                         | 5                              |

Por otro lado tenemos los palm mutes:

   PM----|   PM    PM          PM----|   PM    PM          PM----|   PM    PM          PM----|   PM    PM          PM----|     PM    PM     PM

Tenemos los espaciadores de las tablaturas:

|
|
|
|
|
|

Por otro lado tenemos la tablatura:

---------------------------|---------------------------|---------------------------|---------------------------|--------------------------------|
---------------8-----7--8--|---------------8-----7--8--|---------------8-----7--8--|---------------8-----7--8--|---------10-------12-----10-----|
--------9------------------|--------9------------------|--------9------------------|--------9------------------|------9---------------9------9--|
-----9-----9------9--------|-----9-----9------9--------|-----9-----9------9--------|-----9-----9------9--------|--10---------10-----------------|
--7------------------------|--7------------------------|--7------------------------|--7------------------------|--------------------------------|
---------------------------|---------------------------|---------------------------|---------------------------|--------------------------------|

Donde siempre se dejan dos guiones (-) entre notas que formen parte de la misma agrupación, en este caso por cada compás tenemos 8 corcheas agrupadas en 2 grupos de 4 por 4.
Entre corcheas meteremos dos guiones al final del compas y dos guiones al principio del siguiente compas. Intercalando las barras (|) ya mencionadas entre compás y compás.

Por último y no menos importante, tenemos los tiempos:

    ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½    ½    ½   ½  ½   ½    

Ahi se pueden ver los 2 grupos e corcheas separados por 2 espacios, y luego tres espacios para el siguiente grupo. A excepcion de entre compas y compas que hay 5 espacios, 2 para finalizar el compas, 1 para la bara y otros dos espacios para empezar el siguiente compás

A todo esto, hay que tener en cuenta tanto a la hora de generar la tablaura, como los tiempos de cada figura, que si hay un número de dos cifrás, posiblemente se desalinee el texto. Por ello presta especial atención a esos casos.

En definitiva el texto debe ir bien alineado, tanto como para los palm mutes, como para el numero compas, como las notas en sí, los tiempos de cada figura, los guiones entre nota y nota dependiendo de las agrupaciones (que en este caso son corcheas, pero nos pueden venir negras, blancas, redondas, semicorcheas, fusas, semifusas...) 