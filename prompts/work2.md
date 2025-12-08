Habiendo identificado correctamente cada figura el siguiente paso será usar el txt (examples/validation.txt) donde te explicré a continuación el resultado el equivalente en texto al que quiero que conviertas dicha imagen.

Ahora mismo ocr-tab.py genera lo iguiente:

. = 180
PM     |     PM     PM            PM     |     PM     PM            PM     |     P.M     P.M            P.M     [     PM     PM            P M     |     PM     PM      P M
4                    8        7 8                         8        7 8                         8        7 8                         8        7 8               10       12       10
9                                      9                                      9                                      9                                  9                 9        9
9        9        9                    9        9        9                    9        9        9                    9        9        9               10            10
7                                      7                                      7                                      7

Procedo a explicarte como quiero que generes el txt, y así saber tu que debes modificar en ocr-tab.py

Como se puede ver, tenemos metadatos de la información de la tablatura:

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

Y aquí viene lo más importante de la tablatura, a esto dedicale especial atención a la hora de generar el txt, ya que aquí será donde un buen alineamiento hará que se muestre correctamente el txt.

Donde siempre se dejan dos guiones (-) entre notas que formen parte de la misma agrupación, en este caso por cada compás tenemos 8 corcheas agrupadas en 2 grupos de 4 por 4.
Entre corcheas meteremos dos guiones al final del compas y dos guiones al principio del siguiente compas. Intercalando las barras (|) ya mencionadas entre compás y compás.

Por último y no menos importante, tenemos los tiempos:

    ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½  ½   ½  ½  ½  ½     ½  ½  ½    ½    ½   ½  ½   ½    

Ahi se pueden ver los 2 grupos e corcheas separados por 2 espacios, y luego tres espacios para el siguiente grupo. A excepcion de entre compas y compas que hay 5 espacios, 2 para finalizar el compas, 1 para la bara y otros dos espacios para empezar el siguiente compás

A todo esto, hay que tener en cuenta tanto a la hora de generar la tablaura, como los tiempos de cada figura, que si hay un número de dos cifrás, posiblemente se desalinee el texto. Por ello presta especial atención a esos casos.

En definitiva el texto debe ir bien alineado, tanto como para los palm mutes, como para el numero compas, como las notas en sí, los tiempos de cada figura, los guiones entre nota y nota dependiendo de las agrupaciones (que en este caso son corcheas, pero nos pueden venir negras, blancas, redondas, semicorcheas, fusas, semifusas...) 