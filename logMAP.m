function discriminant = logMAP(x, c, mv, Bh)

z = Bh * (x - mv);   % Bh=B' 
discriminant = c - z' * z;