# -*- coding: utf-8 -*-
"""
Realizovany vs. Nerealizovany zisk
- Dani se pouze realizovany zisk (kdyz prodame akcii, tak rozdil mezi nakupni a prodejni cenou)

Akcie Vyhody:
- Casovy test: od nakupu akcie ubehne doba 3 let, tak se nemusi platit dan ze zisku (FIFO)
- Objem vsech obchodovanych prostredku (trzka: je pouze prijem?) je mensi, jak 100 000 Kc, tak se
    nemusi platit dan ze zisku
- Optimalizace portfolio: Udelam ztratu (prodam ztratovou akcii, tu mohu potom hned nakoupit) a mam ztratu (hodnota?)


Dividendy obecne:
- Vyhody:
- - Do 6000 Kc rocnich dividend se nemusi platit dan ze zisku (ani udavat do danoveho priznani)

Dividendy z CR:
- Neudavame do danoveho priznani, ikdyz nam broker strhne dan a dividenty presahnou 6000 Kc.

Dividendy ze zahranici (ze spolecnosti akcii):
- Vzdy udavame do danoveho priznani, ikdyz nam broker strhne dan. (jen pokud to presahne 6000 Kc?)

Traiding: Derivaty: CFD, ...:

Jine: Investtown, ...:
- Zalezi na kazde spolecnosti, proto je potreba se zeptat, protoze pokdu ne je povinnost na me

Odpisy z dani:
- Naklad na poplatek formou spreadu brokerovi

Chyby:
- Nelze michat zisky a ztraty mezi: Kryptpmenama, akcemi, derivaty, nemovitosti (pronajem), ...
    jsou to ruzne druhy prijmu
- Pri presazeni mnozstevniho testu 100 000 Kc, se musi podat danove prizani ikdyz bud v minusu,
    sice nebudu platit dane, ale mam povinnost to oznamit

Frakcni akcie:
- neni to akcie pokud to neni alespon 1 celek (bere se to neco, jako derivat?),
    ale pokud to presahuje 1 celek (1.1; 1.688; ...), tak se to uz bere jako akcie

"""


class Tax:
    def __init__(self, name, tax_rate):
        self.name = name
        self.tax_rate = tax_rate

    def get_tax_of_profit(self):
        raise NotImplementedError

    def get_tax_of_invest_o_trade(self):
        raise NotImplementedError

    def set_tax_rate(self, tax_rate):
        self.tax_rate = tax_rate
