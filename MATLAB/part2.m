% Benchmark Functions Comparison

% Define test functions
functions = {@(x) sum(x.^2), ... % Sphere function
             @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2), ... % Rosenbrock function
             @(x) -20*exp(-0.2*sqrt(sum(x.^2)/numel(x))) - exp(sum(cos(2*pi*x))/numel(x)) + 20 + exp(1)}; % Ackley function

function_names = {'Sphere', 'Rosenbrock', 'Ackley'};

% Dimensions to test
dimensions = [2, 10];

% Number of runs
num_runs = 15;

% Optimization algorithms to compare
algorithms = {@simple_ga, @simple_pso, @simple_sa};
alg_names = {'Simple Genetic Algorithm', 'Simple Particle Swarm Optimization', 'Simple Simulated Annealing'};

% Results storage
results = struct();

% Convergence history storage
convergence_history = cell(length(dimensions), length(functions), length(algorithms));

% Create 3D surface plots for each function
create_3d_plots(functions, function_names);

for d = 1:length(dimensions)
    D = dimensions(d);
    
    for f = 1:length(functions)
        func = functions{f};
        func_name = function_names{f};
        
        % Set bounds for optimization
        lb = -100 * ones(1, D);
        ub = 100 * ones(1, D);
        
        for a = 1:length(algorithms)
            alg = algorithms{a};
            alg_name = alg_names{a};
            
            fprintf('Testing %s on %s (D=%d)\n', alg_name, func_name, D);
            
            % Run optimization multiple times
            fitness_values = zeros(1, num_runs);
            for run = 1:num_runs
                [~, fval, history] = alg(func, D, lb, ub);
                fitness_values(run) = fval;
                
                % Store convergence history for the first run
                if run == 1
                    convergence_history{d, f, a} = history;
                end
            end
            
            % Store results
            results(d,f,a).algorithm = alg_name;
            results(d,f,a).function = func_name;
            results(d,f,a).dimension = D;
            results(d,f,a).mean = mean(fitness_values);
            results(d,f,a).std = std(fitness_values);
            results(d,f,a).best = min(fitness_values);
            results(d,f,a).worst = max(fitness_values);
        end
    end
end

% Display results
for d = 1:length(dimensions)
    for f = 1:length(functions)
        fprintf('\nResults for %s (D=%d):\n', function_names{f}, dimensions(d));
        for a = 1:length(algorithms)
            r = results(d,f,a);
            fprintf('%s:\n', r.algorithm);
            fprintf('  Mean: %.4e (±%.4e)\n', r.mean, r.std);
            fprintf('  Best: %.4e\n', r.best);
            fprintf('  Worst: %.4e\n', r.worst);
        end
    end
end

% Plot convergence
for d = 1:length(dimensions)
    for f = 1:length(functions)
        figure;
        hold on;
        for a = 1:length(algorithms)
            history = convergence_history{d, f, a};
            plot(1:length(history), history, 'DisplayName', alg_names{a});
        end
        xlabel('Iterations');
        ylabel('Fitness Value');
        title(sprintf('Convergence for %s (D=%d)', function_names{f}, dimensions(d)));
        legend('show');
        set(gca, 'YScale', 'log');  % Use log scale for y-axis
        hold off;
    end
end

% Function to create 3D surface plots
function create_3d_plots(functions, function_names)
    [X, Y] = meshgrid(-5:0.1:5, -5:0.1:5);
    for i = 1:length(functions)
        Z = zeros(size(X));
        for j = 1:size(X, 1)
            for k = 1:size(X, 2)
                Z(j, k) = functions{i}([X(j, k), Y(j, k)]);
            end
        end
        figure;
        surf(X, Y, Z);
        title(['3D Surface Plot of ', function_names{i}, ' Function']);
        xlabel('x');
        ylabel('y');
        zlabel('z');
    end
end

% Simple Genetic Algorithm
function [best_solution, best_fitness, history] = simple_ga(func, D, lb, ub)
    pop_size = 50;
    num_generations = 100;
    mutation_rate = 0.1;
    
    population = lb + (ub - lb) .* rand(pop_size, D);
    fitness = zeros(pop_size, 1);
    history = zeros(1, num_generations);
    
    for gen = 1:num_generations
        for i = 1:pop_size
            fitness(i) = func(population(i, :));
        end
        
        [best_fitness, idx] = min(fitness);
        best_solution = population(idx(1), :);
        history(gen) = best_fitness;
        
        % Select top 2 individuals (or clone the best if only one unique fitness value)
        [~, sorted_idx] = sort(fitness);
        if length(unique(fitness)) == 1
            new_pop = repmat(population(sorted_idx(1), :), 2, 1);
        else
            new_pop = population(sorted_idx(1:2), :);
        end
        
        while size(new_pop, 1) < pop_size
            parents = population(randperm(pop_size, 2), :);
            child = mean(parents, 1);
            if rand < mutation_rate
                child = child + randn(1, D) .* (ub - lb) * 0.1;
            end
            child = max(min(child, ub), lb);  % Ensure child is within bounds
            new_pop = [new_pop; child];
        end
        population = new_pop;
    end
end

% Simple Particle Swarm Optimization
function [best_solution, best_fitness, history] = simple_pso(func, D, lb, ub)
    num_particles = 50;
    num_iterations = 100;
    w = 0.7;
    c1 = 1.4;
    c2 = 1.4;
    
    positions = lb + (ub - lb) .* rand(num_particles, D);
    velocities = zeros(num_particles, D);
    personal_best_pos = positions;
    personal_best_fit = inf(num_particles, 1);
    global_best_pos = zeros(1, D);
    global_best_fit = inf;
    history = zeros(1, num_iterations);
    
    for iter = 1:num_iterations
        for i = 1:num_particles
            fitness = func(positions(i, :));
            if fitness < personal_best_fit(i)
                personal_best_fit(i) = fitness;
                personal_best_pos(i, :) = positions(i, :);
            end
            if fitness < global_best_fit
                global_best_fit = fitness;
                global_best_pos = positions(i, :);
            end
        end
        
        history(iter) = global_best_fit;
        
        for i = 1:num_particles
            r1 = rand(1, D);
            r2 = rand(1, D);
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * r1 .* (personal_best_pos(i, :) - positions(i, :)) + ...
                c2 * r2 .* (global_best_pos - positions(i, :));
            positions(i, :) = positions(i, :) + velocities(i, :);
            positions(i, :) = max(min(positions(i, :), ub), lb);
        end
    end
    best_solution = global_best_pos;
    best_fitness = global_best_fit;
end

% Simple Simulated Annealing
function [best_solution, best_fitness, history] = simple_sa(func, D, lb, ub)
    max_iter = 1000;
    T0 = 100;
    Tf = 1e-8;
    alpha = 0.95;
    
    current_solution = lb + (ub - lb) .* rand(1, D);
    current_energy = func(current_solution);
    best_solution = current_solution;
    best_fitness = current_energy;
    T = T0;
    history = zeros(1, max_iter);
    
    for i = 1:max_iter
        neighbor = current_solution + randn(1, D) .* (ub - lb) * 0.1;
        neighbor = max(min(neighbor, ub), lb);
        neighbor_energy = func(neighbor);
        
        if neighbor_energy < current_energy || rand < exp((current_energy - neighbor_energy) / T)
            current_solution = neighbor;
            current_energy = neighbor_energy;
        end
        
        if current_energy < best_fitness
            best_solution = current_solution;
            best_fitness = current_energy;
        end
        
        history(i) = best_fitness;
        
        T = T0 * alpha^i;
        if T < Tf
            break;
        end
    end
    history = history(1:i);  % Trim unused elements if SA terminates early
end
