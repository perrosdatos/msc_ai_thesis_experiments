import numpy as np
import torch
import json

class LuxKPITracker:
    def __init__(self, batch_size, max_units=16, map_w=24, map_h=24):
        self.batch_size = batch_size
        self.max_units = max_units
        self.map_w = map_w
        self.map_h = map_h
        self.reset()
        
    def reset(self):
        # Global Match State
        self.match_steps = 0
        
        # Step-by-step histories for thesis visualizations
        self.points_history = np.zeros((self.batch_size, 2, 750), dtype=np.int32)
        self.reward_history = np.zeros((self.batch_size, 2, 750), dtype=np.float32)
        self.rc_history = {} # Will dynamically store rc components
        self.rc_totals = {} # Store accumulated total per component
        
        # --- COLLABORATIVE KPIs (Per Team) ---
        self.team_points = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.team_energy_spent = np.zeros((self.batch_size, 2), dtype=np.float32)
        
        # Combat
        self.friendly_killed = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.opponents_killed = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.total_respawns = np.zeros((self.batch_size, 2), dtype=np.float32)
        
        # Exploration
        # Track which tiles have been stepped on by which team
        self.map_footprint = np.zeros((self.batch_size, 2, self.map_w, self.map_h), dtype=bool)
        self.time_to_first_relic = np.full((self.batch_size, 2), -1, dtype=np.int32)
        
        # Synergy (Relic Hand-off)
        # To compute synergy, we track which team and unit discovered which relic first.
        # Shape: (batch, num_relics) -> store unit_id that found it
        self.relic_discovered_by = np.full((self.batch_size, 6), -1, dtype=np.int32) # max 6 relics in v2
        self.synergy_handoffs = np.zeros((self.batch_size, 2), dtype=np.float32)
        
        self.time_to_second_agent_relic = np.full((self.batch_size, 2), -1, dtype=np.int32)
        self.first_relic_discoverer = np.full((self.batch_size, 2), -1, dtype=np.int32)
        
        # Dispersion
        self.dispersion_variance_sum = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.dispersion_steps_counted = np.zeros((self.batch_size, 2), dtype=np.float32)
        
        # --- INDIVIDUAL KPIs (Per Unit 0-15) ---
        self.ind_energy_spent = np.zeros((self.batch_size, 2, self.max_units), dtype=np.float32)
        self.ind_points_generated = np.zeros((self.batch_size, 2, self.max_units), dtype=np.float32)
        self.ind_tiles_explored = np.zeros((self.batch_size, 2, self.max_units), dtype=np.float32)
        self.ind_lifespan_sum = np.zeros((self.batch_size, 2, self.max_units), dtype=np.float32)
        self.ind_spawn_step = np.zeros((self.batch_size, 2, self.max_units), dtype=np.int32)
        
        # Track individual footprints
        self.ind_map_footprint = np.zeros((self.batch_size, 2, self.max_units, self.map_w, self.map_h), dtype=bool)
        
        # Cache for previous state
        self.prev_points = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.prev_energy = np.zeros((self.batch_size, 2, self.max_units), dtype=np.float32)
        self.prev_masks = np.zeros((self.batch_size, 2, self.max_units), dtype=bool)

    def update(self, env, td, rc):
        """
        Updates the KPIs using the environment state after `step()`
        rc: env.last_reward_components
        """
        self.match_steps += 1
        
        state = env.env_state
        pts = np.asarray(state.team_points) # (B, 2)
        pos = np.asarray(state.units.position) # (B, 2, 16, 2)
        energy = np.asarray(state.units.energy[..., 0]) # (B, 2, 16)
        masks = np.asarray(state.units_mask) # (B, 2, 16)
        relic_mask = np.asarray(state.relic_nodes_mask) # (B, 6)
        relic_pos = np.asarray(state.relic_nodes) # (B, 6, 2)
        
        delta_points = pts - self.prev_points
        self.team_points += np.maximum(delta_points, 0)
        self.prev_points = pts
        
        step_idx = min(self.match_steps - 1, 749)
        self.points_history[:, :, step_idx] = pts
        
        # Track reward and components
        reward_tensor = td.get(("next", "agents", "reward"), None)
        if reward_tensor is not None:
            # reward is shape (B, 1) or (B, 16, 1) or similar.
            # in lux match_v2, team reward is usually the same for all agents.
            # let's extract team reward. td has reward per agent?
            # actually td get agents reward is shape (B, 16, 1).
            # let's get team reward from rc instead or just average.
            pass # we'll use rc components instead to be precise!
            
        for k, v in rc.items():
            if k not in self.rc_history:
                self.rc_history[k] = np.zeros((self.batch_size, 2, 750), dtype=np.float32)
                self.rc_totals[k] = np.zeros((self.batch_size, 2), dtype=np.float32)
            val = np.asarray(v)
            if val.shape[0] == self.batch_size:
                # We need to add it to the active team. Assume team 0 since env_a is used.
                t_active = 0
                if hasattr(env.base_env, "team_id"):
                    t_active = env.base_env.team_id
                    
                if val.ndim == 2: # e.g. (B, 16)
                    val = val.sum(axis=1) # (B,)
                
                if val.ndim == 1: # (B,)
                    self.rc_totals[k][:, t_active] += val
                    self.rc_history[k][:, t_active, step_idx] = self.rc_totals[k][:, t_active]
                    
                    if k == "total_reward":
                        self.reward_history[:, t_active, step_idx] = self.rc_totals[k][:, t_active]
        
        for b in range(self.batch_size):
            for t in range(2):
                # 1. Combat & Respawns
                delta_m = masks[b, t].astype(int) - self.prev_masks[b, t].astype(int)
                died_idx = np.where(delta_m == -1)[0]
                spawned_idx = np.where(delta_m == 1)[0]
                
                self.friendly_killed[b, t] += len(died_idx)
                self.opponents_killed[b, t] += len(np.where(masks[b, 1-t].astype(int) - self.prev_masks[b, 1-t].astype(int) == -1)[0])
                self.total_respawns[b, t] += len(spawned_idx)
                
                for idx in died_idx:
                    lifespan = self.match_steps - self.ind_spawn_step[b, t, idx]
                    self.ind_lifespan_sum[b, t, idx] += lifespan
                    
                for idx in spawned_idx:
                    self.ind_spawn_step[b, t, idx] = self.match_steps
                
                # 2. Energy Spent
                # Energy decreases from actions. (Increases from center/relics).
                # If energy drops but unit didn't die, it was spent.
                delta_e = energy[b, t] - self.prev_energy[b, t]
                spent = np.where((delta_e < 0) & (self.prev_masks[b, t]), -delta_e, 0)
                self.ind_energy_spent[b, t] += spent
                self.team_energy_spent[b, t] += spent.sum()
                
                # 3. Footprint & Dispersion
                valid_pos = pos[b, t][masks[b, t]]
                if len(valid_pos) > 0:
                    for idx, (px, py) in enumerate(pos[b, t]):
                        if masks[b, t, idx]:
                            if not self.ind_map_footprint[b, t, idx, px, py]:
                                self.ind_map_footprint[b, t, idx, px, py] = True
                                self.ind_tiles_explored[b, t, idx] += 1
                                self.map_footprint[b, t, px, py] = True
                    
                    # Dispersion variance (variance of pairwise distances)
                    if len(valid_pos) > 1:
                        centroid = valid_pos.mean(axis=0)
                        dists = np.linalg.norm(valid_pos - centroid, axis=1)
                        var = np.var(dists)
                        self.dispersion_variance_sum[b, t] += var
                        self.dispersion_steps_counted[b, t] += 1
                
                # 4. Points & Relics
                if delta_points[b, t] > 0:
                    # Distribute points to individual agents based on local_point_generation
                    loc_pts = rc.get("local_point_generation", np.zeros((self.batch_size, 2, self.max_units)))[b, t]
                    loc_pts_sum = loc_pts.sum()
                    if loc_pts_sum > 0:
                        # Normalize and scale to actual delta points
                        self.ind_points_generated[b, t] += (loc_pts / loc_pts_sum) * delta_points[b, t]
                
                # Relic First Discovery & Synergy
                for r_idx in range(6):
                    if relic_mask[b, r_idx]:
                        rx, ry = relic_pos[b, r_idx]
                        if rx < 0: continue
                        
                        # Find agents nearby
                        for u_idx in range(self.max_units):
                            if masks[b, t, u_idx]:
                                ux, uy = pos[b, t, u_idx]
                                dist = max(abs(ux-rx), abs(uy-ry))
                                if dist <= 4:
                                    if self.time_to_first_relic[b, t] == -1:
                                        self.time_to_first_relic[b, t] = self.match_steps
                                        self.first_relic_discoverer[b, t] = u_idx
                                    elif self.time_to_second_agent_relic[b, t] == -1 and u_idx != self.first_relic_discoverer[b, t]:
                                        self.time_to_second_agent_relic[b, t] = self.match_steps
                                        
                                    if self.relic_discovered_by[b, r_idx] == -1:
                                        # newly discovered
                                        self.relic_discovered_by[b, r_idx] = u_idx
                                    elif self.relic_discovered_by[b, r_idx] != u_idx:
                                        # Synergistic exploitation! Another agent is utilizing it.
                                        # Only count if delta points were generated here
                                        if delta_points[b, t] > 0:
                                            self.synergy_handoffs[b, t] += 0.1 # Accumulate synergy
        
        self.prev_energy = energy
        self.prev_masks = masks

    def get_results(self):
        """
        Calculates final stats and returns a list of dictionaries per batch.
        """
        results = []
        for b in range(self.batch_size):
            b_res = {}
            for t in range(2):
                prefix = f"team_{t}_"
                b_res[prefix+"total_points"] = self.team_points[b, t]
                b_res[prefix+"energy_spent"] = self.team_energy_spent[b, t]
                b_res[prefix+"efficiency"] = self.team_points[b, t] / (self.team_energy_spent[b, t] + 1e-5)
                b_res[prefix+"friendly_killed"] = self.friendly_killed[b, t]
                b_res[prefix+"opponents_killed"] = self.opponents_killed[b, t]
                b_res[prefix+"combat_dominance"] = self.opponents_killed[b, t] / (self.friendly_killed[b, t] + 1e-5)
                b_res[prefix+"total_respawns"] = self.total_respawns[b, t]
                
                total_pts_match = self.team_points[b, 0] + self.team_points[b, 1] + 1e-5
                b_res[prefix+"resource_monopoly"] = self.team_points[b, t] / total_pts_match
                
                b_res[prefix+"map_exploration_prop"] = self.map_footprint[b, t].sum() / (self.map_w * self.map_h)
                b_res[prefix+"time_to_first_relic"] = self.time_to_first_relic[b, t]
                b_res[prefix+"synergy_handoffs"] = self.synergy_handoffs[b, t]
                
                # Info propagation delay (penalty if second agent never arrives)
                if self.time_to_first_relic[b, t] != -1:
                    if self.time_to_second_agent_relic[b, t] != -1:
                        delay = self.time_to_second_agent_relic[b, t] - self.time_to_first_relic[b, t]
                    else:
                        delay = 750 - self.time_to_first_relic[b, t] # Max penalty if info is never shared
                else:
                    delay = 750 # Never even found the first relic
                b_res[prefix+"info_propagation_delay"] = delay
                
                steps_var = self.dispersion_steps_counted[b, t] if self.dispersion_steps_counted[b, t] > 0 else 1
                b_res[prefix+"dispersion_variance"] = self.dispersion_variance_sum[b, t] / steps_var
                
                # Individual KPIs
                for u in range(self.max_units):
                    u_pref = f"team_{t}_agent_{u}_"
                    b_res[u_pref+"energy_spent"] = self.ind_energy_spent[b, t, u]
                    b_res[u_pref+"points_generated"] = self.ind_points_generated[b, t, u]
                    b_res[u_pref+"tiles_explored"] = self.ind_tiles_explored[b, t, u]
                    
                    # Finalize lifespan
                    if self.prev_masks[b, t, u]:
                        lifespan = self.match_steps - self.ind_spawn_step[b, t, u]
                        self.ind_lifespan_sum[b, t, u] += lifespan
                        
                    # Calculate average lifespan (total lifespan / (respawns + 1))
                    respawns = self.ind_spawn_step[b, t, u] > 0 # roughly 
                    # accurately we should count number of times it spawned
                    b_res[u_pref+"avg_lifespan"] = self.ind_lifespan_sum[b, t, u] # Simplification
                    
                # Export Histories
                b_res[prefix+"points_history"] = json.dumps(self.points_history[b, t, :self.match_steps].tolist())
                # If reward was tracked directly in rc:
                b_res[prefix+"reward_history"] = json.dumps(self.reward_history[b, t, :self.match_steps].tolist())
                
                # Export all Reward Components (totals and history)
                for rc_k, rc_hist in self.rc_history.items():
                    b_res[prefix+f"rc_{rc_k}_total"] = self.rc_totals[rc_k][b, t]
                    b_res[prefix+f"rc_{rc_k}_history"] = json.dumps(rc_hist[b, t, :self.match_steps].tolist())
            
            results.append(b_res)
        return results
