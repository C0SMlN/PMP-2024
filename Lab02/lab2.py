import random

bile_rosii = 3
bile_albastre = 4
bile_negre = 2

def simulare():
    bile_rosii_extrase = 0

    for i in range(10000):
        urna = ["rosu"] * bile_rosii +  ["albastru"] * bile_albastre + ["negru"] * bile_negre

        diceroll = random.randint(1, 6)
        if diceroll in [2, 3, 5]:
            urna.append("negru")
        elif diceroll == 6:
            urna.append("rosu")
        else:
            urna.append("albastru")

        extragere = random.choice(urna)
        print(len(urna))
        urna.remove(extragere)
        print(len(urna))
        exit()

        if extragere == "rosu":
            bile_rosii_extrase += 1
    probabilitatea_est = bile_rosii_extrase / 10000

    return probabilitatea_est

print(simulare())