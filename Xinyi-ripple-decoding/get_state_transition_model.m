function [state_transition_model] = get_state_transition_model(state_index, linear_distance_bins, linear_distance)
%% Whether the replay path follows an outbound or inbound movement trajectory
% empirical movement transition matrix conditioned on discrete state
num_discrete_states = length(state_index);
empirical_movement_transition_matrix = cell(num_discrete_states, 1);
for state_number = 1:num_discrete_states,
    empirical_movement_transition_matrix{state_number} = condition_empirical_movement_transition_matrix_on_state(linear_distance_bins, linear_distance, state_index{state_number});
end

state_transition_model{1} = empirical_movement_transition_matrix{1}; % outbound forward
state_transition_model{2} = empirical_movement_transition_matrix{2}; % outbound reverse, same as inbound forward
state_transition_model{3} = empirical_movement_transition_matrix{2}; % inbound forward
state_transition_model{4} = empirical_movement_transition_matrix{1}; % inbound reverse, same as outbound forward
end