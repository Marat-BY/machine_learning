import numpy as np 
impot pandas as pd 
# Starting point position on coordiantes
start_position = [
					[1, 2, 3, 4],
					[5, 6, 7, 8],
					[9, 10, 11, 12],
					[13, 14, 15, 16],
				  ]
# Ending point position on coordinates
end_position = [
					[1, 2, 3, 4],
					[5, 6, 7, 8],
					[9, 10, 11, 12],
					[13, 0, 14, 15],
				  ]

# Forming a dictionary from coordinates planes
def dictFrompositions(position):
	"""
	   Формирование словаря из координат чисел,
	   где ключ словаря - номер позиции
	       значения - координаты х и у
	"""
	position_dict = {}

	row_number = 1
	for row in position:

		column_number = 1
		for column in row:
			if column != 0:
				position_dict[column] = [row_number, column_number]

			column_number = column_number + 1

		row_number = row_number + 1

	return position_dict

# Calculating a Manhattan Distance Metric:
start_position = dictFrompositions(start_position)
print(start_position)

end_position = dictFrompositions(end_position)
print(start_position)

# To calculate MD formula is absolut distance between 2 points
# SUM(|X - Y|)
for key, coordinates in start_position.items():
	# print (key, 'coordinates', 'Manhattan Distance')
	print('(X2 - X1):', start_position[key][0] - end_position[key][0], 
          '(Y2 - Y1):', start_position[key][1] - end_position[key][1],
          'Manhattan distaince: ', 
          abs((abs(start_position[key][0] - end_position[key][0])) - (abs(start_position[key][1] - end_position[key][1]))))
