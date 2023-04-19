Dataset from : https://github.com/oracle-devrel/redbull-pit-strategy/

Ziel:

Es wird versucht die Stint länge (die Zeit auf der Rennstrecken zwischen den Pit-Stops) so groß wie mögich zu halten. Im Umkehrschluss verringert sich die Anzahl der Stints/designedLaps

Zu untersuchende Features:
    hist:
        Compound
        Stint
        bestPreRaceTime, hue=bestLapTimeIsFrom
        Compound / Rainfall
        Compound / Stint
        Compound / StintLen
        Compound / meanTrackTemp
    
    lmplot:
        CircuitLength / StintLen
        designedLaps/ StintLen

    boxplot:
        stint





