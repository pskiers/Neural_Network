<div style="padding: 2% 5%;">

<h1 style="text-align: center;">
<div style="color:grey; font-size: 0.6em;">Jakub Ostrzołek, Paweł Skierś</div>
<div>WSI ćwiczenie 5 - sieci neuronowe</div>
</h1>

## Opis ćwiczenia
Celem ćwiczenia było zaimplementowanie sieci neuronowych. 

Klasa implementująca drzewo ma jeden parametr konstruktora:
* `max_depth` - maksymalna wysokość drzewa powstałego w wyniku uczenia się.

Klasa ta ma 2 główne funkcje (zgodne z biblioteką sklearn):
* `fit` - uczenie drzewa na podstawie danych wejściowych i przypisanych im klas wyjściowych,
* `predict` - przewidywanie klas dla danych wejściowych (wcześniej nauczonych przez `fit`).

## Wykorzystane zewnętrzne biblioteki
* `numpy`
* `pandas`
* `matplotlib`
* `sklearn`

## Trenowanie drzewa
Aby wytrenować drzewo należy wykonać skrypt `main.py` i postępować zgodnie z instrukcjami (`main.py --help`).  
Skrypt wygeneruje nowe drzewo, wytrenuje je na podstawie danych ze zbioru danych _iris_, oraz pokaże osiągi drzewa w postaci wybranych metryk i macierzy konfuzji na wykresie.
 
## Wykresy
Aby wygenerować wykresy, należy wykonać skrypt `plot.py`. Za pomocą skryptu można wygenerować:
* macierze konfuzji dla wybranych zbiorów danych
* porównanie metryk dla wybranych zbiorów danych
* powyższe wykresy dla najlepszej kombinacji hiperparametrów (pod względem sumy metryk)

Każdy wykres jest opisany dwoma hiperparametrami:
* `depth` - maksymalna głębokość drzewa
* `n_bins` - "rozdzielczość" dyskretyzacji danych wejściowych (dane wejściowe są grupowane w klasy tak, by w każdej klasie była porównywalna ilość rekordów)

Oto przykładowe wyniki:
* `n_bins = 5` - różne maksymalne głębokości drzewa

![wykres](plots/b=5&d=0.jpg)
![wykres](plots/b=5&d=1.jpg)
![wykres](plots/b=5&d=2.jpg)
![wykres](plots/b=5&d=3.jpg)
![wykres](plots/b=5&d=4.jpg)
* `depth = 5` - różne "rozdzielczości" dyskretyzacji danych

![wykres](plots/b=2&d=4.jpg)
![wykres](plots/b=3&d=4.jpg)
![wykres](plots/b=5&d=4.jpg)
![wykres](plots/b=7&d=4.jpg)

## Wnioski

1. Overfitting
2. Underfitting
3. Batch size
4. learning rate

</div>