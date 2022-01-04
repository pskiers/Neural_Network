<div style="padding: 2% 5%;">

<h1 style="text-align: center;">
<div style="color:grey; font-size: 0.6em;">Jakub Ostrzołek, Paweł Skierś</div>
<div>WSI ćwiczenie 5 - sieci neuronowe</div>
</h1>

## Opis ćwiczenia
Celem ćwiczenia było zaimplementowanie sieci neuronowych. 

Klasa implementująca warstę sieci przyjmuje następujące parametry konstruktora:
* `input_size` - wymiar wektora wejściowego
* `output_size` - wymiar wektora wyjściowego
* `activation` - funkcja aktywacji
* `activation_grad` - gradient funkcji aktywacji
* `output_inicialization` - czy inicjalizować wagi zerami (w przeciwnym wypadku inicjalizuje losowo zgodnie z optymalnym rozkładem), powinien być ustawiony na `True` w ostatniej warstwie

Klasa implementująca sieć przyjmuje następujące parametry konstruktora:
* `is_classifier` - jeżeli sieć jest klasyfikatorem, to dla każdej epoki jest obliczana również dokładność przewidywania.
* `layers...` - warstwy sieci (można dodać również do istniejącej sieci za pomocą metody `add_layer`)

Sieć posiada funkcje `fit` i `predict`, służące odpowiednio do trenowania i przewidywania, działające zgodnie z modelami z biblioteki `sklearn`.

## Wykorzystane zewnętrzne biblioteki
* `numpy`
* `pandas`
* `matplotlib`
* `sklearn`

## Testowanie sieci
Aby przetestowyać sieć należy wykonać skrypt `main.py`, uprzednio zmieniając jej parametry zgodnie z zapotrzebowaniem.  
Skrypt wygeneruje nową sieć, wytrenuje ją na podstawie danych ze zbioru _minist_, oraz pokaże wykresy przedstawiające historię trenowania sieci oraz jej osiągi w postaci metryk i macierzy konfuzji.
 
## Wykresy i wnioski


### Batch size
batch size | historia | metryki
-|-|-
8 | ![wykres](plots/batch_size/history,layers=[512,256,128,64],batch_size=8,learn_rate=0.01,epochs=100.png) | ![wykres](plots/batch_size/metrics,layers=[512,256,128,64],batch_size=8,learn_rate=0.01,epochs=100.png)
32 | ![wykres](plots/batch_size/history,layers=[512,256,128,64],batch_size=32,learn_rate=0.01,epochs=100.png) | ![wykres](plots/batch_size/metrics,layers=[512,256,128,64],batch_size=32,learn_rate=0.01,epochs=100.png)
128 | ![wykres](plots/batch_size/history,layers=[512,256,128,64],batch_size=128,learn_rate=0.01,epochs=100.png) | ![wykres](plots/batch_size/metrics,layers=[512,256,128,64],batch_size=128,learn_rate=0.01,epochs=100.png)
512 | ![wykres](plots/batch_size/history,layers=[512,256,128,64],batch_size=512,learn_rate=0.01,epochs=100.png) | ![wykres](plots/batch_size/metrics,layers=[512,256,128,64],batch_size=512,learn_rate=0.01,epochs=100.png)

* im większy batch size, tym szybciej wykonują się epoki (jedna operacja na macierzy jest szybsza niż wiele operacji na jej wierszach, np. dzięki temu, że może zostać użyta jednostka wektorowa; kod z bibliotek może być już skompilowany; wielokrotne wywoływanie funkcji na każdym wierszu jest wolne)
* im mniejszy batch size, tym większa skłonność modelu do przetrenowania (dla większych wartości tego parametru gradient wag jest średnią gradientów wag z większej próby, co lepiej przybliża zbiór walidacyjny / testowy)
* większy batch size poprawia osiągi na zbiorze testowym, ale zbyt duży powoduje spowolnienie uczenia się i pogorsza osiągi.

### Learning rate
learning rate | historia | metryki
-|-|-
0.005 | ![wykres](plots/learning_rate/history,layers=[512,256,128,64],batch_size=128,learn_rate=0.005,epochs=100.png) | ![wykres](plots/learning_rate/metrics,layers=[512,256,128,64],batch_size=128,learn_rate=0.005,epochs=100.png)
0.01 | ![wykres](plots/learning_rate/history,layers=[512,256,128,64],batch_size=128,learn_rate=0.01,epochs=100.png) | ![wykres](plots/learning_rate/metrics,layers=[512,256,128,64],batch_size=128,learn_rate=0.01,epochs=100.png)
0.05 | ![wykres](plots/learning_rate/history,layers=[512,256,128,64],batch_size=128,learn_rate=0.05,epochs=100.png) | ![wykres](plots/learning_rate/metrics,layers=[512,256,128,64],batch_size=128,learn_rate=0.05,epochs=100.png)
0.1 | ![wykres](plots/learning_rate/history,layers=[512,256,128,64],batch_size=128,learn_rate=0.1,epochs=100.png) | ![wykres](plots/learning_rate/metrics,layers=[512,256,128,64],batch_size=128,learn_rate=0.1,epochs=100.png)

* zbyt mały learning rate powoduje, że model się wolniej uczy (wolna eksploracja, duża eksploatacja)
* zbyt duży learning rate powoduje bardziej nieregularne wyniki w uczeniu się modelu, więc trudniej mu znaleźć optimum (szybka eksploracja, mała eksploatacja)
* przekroczenie pewnego progu parametru learning rate powoduje, że model może rozbiegać od rozwiązania

<!-- 
1. Overfitting
2. Underfitting
3. Batch size
4. learning rate
-->

</div>