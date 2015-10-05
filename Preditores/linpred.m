% linpred.m
% EA072 - 2s2015 - Prof. Von Zuben
% Marcelo M Toledo (094139)
% Script containing some auxiliary functions

% Finds the best c regularization factor for a given training and validation set.
% Plots for each iteration
function [best_c, best_b_coefs, best_rEQM, S_valid, best_measured] = find_best_c(cs, train_file, validation_file)

	load(train_file);
	X_train = X;
	S_train = S;

	load(validation_file);
	X_valid = X;
	S_valid = S;

	best_c = 0;
	best_rEQM = 100000;
	best_measured = [];

	n = size(S_valid,1);
	range = [1:n];
	
	for i = 1 : size(cs, 2),

		c = cs(i);
		[b_coefs, expected, measured, rEQM] = linear_prediction(X_train, S_train, X_valid, S_valid, c)

		if rEQM < best_rEQM,
			best_rEQM = rEQM;
			best_c = c;
			best_b_coefs = b_coefs;
			best_measured = measured;
		end

		% disp(sprintf('Iteração: %d', i));
		% aux_plot(i, c, range, expected, measured, rEQM);

	end

endfunction

% Plots a comparation between expected and measured values
function aux_plot(i, c, expected, measured, rEQM)

	figure(i);
	a = horzcat(expected, measured);
	p = plot(a);
	title(sprintf('Validação do preditor linear para c = %f', c));
	xlabel(sprintf('Erro rEQM = %f', rEQM));
	ylabel('Erro quadrático médio');
	legend('Esperado', 'Medido');
	grid;

endfunction

% Returns the values of regularization factor to be tested
function cs = c_values()

	cs = 2 .^ [-24:25];
	cs(end + 1) = 0;

endfunction

function [b_coefs, expected, measured, rEQM] = linear_prediction(X_train, S_train, X_valid, S_valid, c)

	b_coefs = compute_b_coefs(X_train, S_train, c);
	measured = predict_all(X_valid, b_coefs);
	expected = S_valid;
	rEQM = squared_mean_error(measured, expected);

endfunction

% Computes the coeficients of the linear predictor
function b_coefs = compute_b_coefs(A, Y, c)
	
	At = A';
	AtA = At*A;
	lines = size(AtA, 1);
	I = eye(lines);
	cI = c * I;
	AtA_cI = AtA + cI;
	invv = inv(AtA_cI);
	quase_b = invv*At;
	b_coefs = quase_b * Y;

endfunction

% Return a vector of all of the predicted values for
% each entry in S - validation set
function X_pred = predict_all(X, b_coefs)

	X_pred = [];
	for i = 1 : size(X,1),

		x = X(i,:)'
		n = size(x,1);
		x = flipud(x(1:n-1));
		x(n) = 1;
		X_pred = vertcat(X_pred, predict(x, b_coefs));

	end

endfunction

% Predicts a value based on its pasts values, using the predictor's
% calculated coeficients from a "training" set
function x_pred = predict(past_values, b_coefs)

	if size(past_values, 1) ~= (size(b_coefs, 1)),
		error('past_values b_coefs size error');
	end

	% x = flipud(past_values);
	x = past_values;
	n = size(x)
	
	mult = x .* b_coefs;

	x_pred = sum(mult);

endfunction

% Returns the squared mean error from a set of measured values and
% a set of expecte values
function rEQM = squared_mean_error(measured, expected)

	if size(measured,1) ~= size(expected, 1),
		error('measured values differ in size with expected values');
	end

	difference = measured - expected;
	squared_diff = difference .^ 2;

	rEQM = sqrt(sum(squared_diff) / size(difference, 1));

endfunction