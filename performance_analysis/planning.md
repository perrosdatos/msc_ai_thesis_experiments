Tengo tres modelos entrenados durante 6000000 steps. Tengo checkpoints cada 150000 steps, con un total de 40. 
Necesito que uses como homepath de este mini analysius la carpeta /home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis


Crea una carpeta dentro de la carpeta del analisis que se llame html_reports, esta vez todos los reportes dentro de la carpeta tendran el nombre del modelo que representan y no mas timestamps para evitar que cada ves que se corran nuevamente los scripts creen extra de archivos, los reportes dentro de esta carpeta son reportes de resultados finales.

Crea un csv con las llaves usadas para los diferentes environments a usar. Pasaremos cada seed para cada environment. el csv tendra dos columnas, la primera el id del envionrment, 1,2,3,... y la segunda columna un numero aleatorio no repetido entre 0 y 100_000, el csv se generara solo una vez mediante un script en python con una seed al inicio que haga reproducible su generacion cada vez. 



Crea una copia del script y sus python scripts de report_utils/luxai/generate_all_reports.sh. Crea un reporte de la arquitectura de los tres modelos dentro de /home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/models. Estos modelos fueron generados con /home/carlos/Documents/github/msc_ai_thesis_experiments/scripts/luxai/run_sweep_night_biggercnn_6m.sh y /home/carlos/Documents/github/msc_ai_thesis_experiments/scripts/luxai/run_qmix_6m.sh.  

Crearas un mododulo en python que medira las siguientes cosas dentro de cada episodio (compuesto por 3 matches):

Por cada equipo:
	kpis colaborativos:
		puntos totales del episodio
		jugadores asesinados propios
		jugadores asesinados del oponente
		energia gastada
		eficiencia: puntos_totales/energia_invertida
		proporcion del mapa explorada (contando todo el tiempo)
		promedio de la proporcion del mapa ocupada
		steps tomados para encontrar la primera reliquia
		numero de reliquias encontradas
		tiempo que tomo desde que un agente encontro una reliquia para que otro descubra la misma reliquia
		numero promedio de reliquias con agentes cercanos (dentro de su ventana de influencia) en steps con ganancia de bonus
		ratio promedio de agentes encimados 1 significa todos los agentes estan sobre la misma celda, 0 significa todos los agentes ocupan solo una celda
		el episodio fue ganado
		cuantos match gano el equipo 
		
	kpis individuales por agente:
		kpis_agente_i:
			los mismos de los colaboratios pero en individual el objetivo es que podamos verificar si por alguna razon la red aprendio roles, por ejemplo el agente 1 es explorador, el agente 2 es explotador de puntos, el agente 3 siempre viaja en horizontal, el agente 4 en vertical, etc. 




