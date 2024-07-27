import numpy as np
import random

class PunktyOdleglosci:
  """
  Klasa generująca losowe punkty w n-wymiarowej przestrzeni i oblicza zdyskretyzowane odległości między nimi.

  Atrybuty:
    n: Liczba wymiarów przestrzeni.
    liczba_punktow: Liczba punktów do wygenerowania.
    min_odleglosc: Minimalna możliwa odległość.
    max_odleglosc: Maksymalna możliwa odległość.
    krok: Krok dyskretyzacji odległości.
    punkty: Tablica zawierająca współrzędne punktów (liczba_punktow x n).
    odleglosci_zdyskretyzowane: Tablica zawierająca zdyskretyzowane odległości między punktami (liczba_punktow x liczba_punktow).

  Metody:
    generuj(): Generuje punkty i oblicza zdyskretyzowane odległości.
    wyswietl_punkty(): Wyświetla współrzędne punktów.
    wyswietl_odleglosci_zdyskretyzowane(): Wyświetla zdyskretyzowane odległości między punktami.
  """

  def __init__(self, n, liczba_punktow, min_odleglosc, max_odleglosc, krok):
    """
    Konstruktor klasy.

    Args:
      n: Liczba wymiarów przestrzeni.
      liczba_punktow: Liczba punktów do wygenerowania.
      min_odleglosc: Minimalna możliwa odległość.
      max_odleglosc: Maksymalna możliwa odległość.
      krok: Krok dyskretyzacji odległości.
    """
    self.n = n
    self.liczba_punktow = liczba_punktow
    self.min_odleglosc = min_odleglosc
    self.max_odleglosc = max_odleglosc
    self.krok = krok
    self.punkty = None
    self.odleglosci_zdyskretyzowane = None

  def generuj(self, scale):
    """
    Generuje punkty i oblicza zdyskretyzowane odległości.
    """
    # Wygeneruj losowe współrzędne punktów
    self.punkty = np.random.rand(self.liczba_punktow, self.n)*(self.max_odleglosc-self.min_odleglosc) +self.min_odleglosc

    # Oblicz odległości między punktami
    self.odleglosci_zdyskretyzowane = np.zeros((self.liczba_punktow, self.liczba_punktow))
    for i in range(self.liczba_punktow):
      for j in range(i + 1, self.liczba_punktow):
        odleglosc = scale*np.linalg.norm(self.punkty[i] - self.punkty[j])
        odleglosc_zdyskretyzowana = (self.min_odleglosc + int((odleglosc - self.min_odleglosc) / self.krok) * self.krok)
        self.odleglosci_zdyskretyzowane[i, j] = odleglosc_zdyskretyzowana#scale*np.linalg.norm(self.punkty[i] - self.punkty[j])
        self.odleglosci_zdyskretyzowane[j, i] = odleglosc_zdyskretyzowana#scale*np.linalg.norm(self.punkty[i] - self.punkty[j])

  def wyswietl_punkty(self):
    """
    Wyświetla współrzędne punktów.
    """
    if self.punkty is None:
      print("Punkty nie zostały wygenerowane. Proszę użyć metody 'generuj()'.")
      return

    print("Punkty:")
    print(self.punkty)

  def wyswietl_odleglosci_zdyskretyzowane(self):
    """
    Wyświetla zdyskretyzowane odległości między punktami.
    """
    if self.odleglosci_zdyskretyzowane is None:
      print("Odległości zdyskretyzowane nie zostały obliczone. Proszę użyć metody 'generuj()'.")
      return

    print("\nOdległości zdyskretyzowane:")
    print(self.odleglosci_zdyskretyzowane)
  def make_pairs(self,n ):
   
    pairPoints = []
    dists = []
   
    

    for idxA in range(len(self.punkty)):

      pointA = self.punkty[idxA]
     
      Btabel = np.random.choice( len(self.punkty),n )
      for idxB in Btabel:
        
      # prepare a positive pair and update the images and labels
      # lists, respectively
        pairPoints.append([pointA,self.punkty[idxB]])
        dists.append([self.odleglosci_zdyskretyzowane[idxA,idxB]])

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairPoints), np.array(dists))


if __name__ == "__main__":
  # Przykład użycia
  n = 5  # Liczba wymiarów przestrzeni
  liczba_punktow = 10
  # Przykład użycia (kontynuacja)

  min_odleglosc = 0  # Minimalna możliwa odległość
  max_odleglosc = 10  # Maksymalna możliwa odległość
  krok = 0.25  # Krok dyskretyzacji odległości

  # Utwórz obiekt klasy
  punkty_odleglosci = PunktyOdleglosci(n, liczba_punktow, min_odleglosc, max_odleglosc, krok)

  # Wygeneruj punkty i obliczenia odległości
  punkty_odleglosci.generuj()

  # Wyświetl punkty
  punkty_odleglosci.wyswietl_punkty()

  # Wyświetl zdyskretyzowane odległości
  punkty_odleglosci.wyswietl_odleglosci_zdyskretyzowane()
