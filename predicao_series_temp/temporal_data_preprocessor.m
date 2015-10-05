clear all;
format long;
format compact;

% Gerando os dados de treinamento (igualmente espaçados)
% disp('Transformação de conjunto de dados em ');
% disp('N - tamanho do conjunto de dados')

% Pre-processing parameters input
% Data set input file (*.dat)
data_set = input('Temporal series data set file name (use single quotes): ');
data_set = dlmread(data_set);

% Data set size
N = size(data_set)(1)

% Preceding values size
preceding_values_size = input('Preceding values window size (L): ');
L = preceding_values_size;

% Data set normalization
disp('Data set normalization:');
disp('    0 - Don t normalize');
disp('    1 - Normalize');
normalize = input('normalize: ');

% Data set normalization
if normalize == 1,
	data_set_max_val = max(data_set);
	data_set = data_set ./ data_set_max_val;
end

% Predictor input matrix and output matrix creation
for i = 1 : (N - L),
	for j = 1 : L,
		X(i, j) = data_set(i + j - 1);
	end
	S(i, 1) = data_set(i + L);
end

% Saving Pre-processed data set
output_file = input('Output file name: ');
save pre_pross_data X S;