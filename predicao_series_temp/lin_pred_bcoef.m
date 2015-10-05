
% 1;

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

function x_pred = predict(past_values, b_coefs)

	if size(past_values, 1) ~= (size(b_coefs, 1) - 1),
		error('past_values b_coefs size error');
	end

	x = flipud(past_values);
	n = size(x)
	x(n+1) = 1;

	mult = x .* b_coefs;

	x_pred = sum(mult);

endfunction