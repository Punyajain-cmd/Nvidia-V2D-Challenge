from pathlib import Path
p = Path("deploy/modularity/bringup.py")
t = p.read_text(encoding="utf-8")

# Fix lift branch: apply width/force to current cmd/_last_cmd, not a stale last
old_lift = '''                    elif phase == "lift" and last is not None:
                        # Keep closed; nudge joints slightly "up"
                        q = self.runtime.adapter.current_joints().copy()
                        if fault == "joint_limit" and len(q) > 0:
                            # Drive near a caricature joint limit for diagnosis.
                            q[0] = float(q[0]) + 1.2
                        elif len(q) > 1:
                            q[1] = float(q[1]) - 0.04  # shoulder lift proxy
                        self.runtime.adapter.set_joints(q)
                        gw_lift = float(getattr(last, "gripper_width", 0.01))
                        if fault == "grasp_slip":
                            # Blended slip: reopen on lift (hist min shows prior close).
                            gw_lift = 0.08
                        elif fault == "premature_open":
                            # Clear policy/cmd open at lift after a real close.
                            gw_lift = 0.08
                            last.gripper_force = 20.0
                        elif fault == "soft_grasp":
                            # Stay near-closed with slight drift; contact still weak.
                            gw_lift = 0.012
                            last.gripper_force = 1.5
                            if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                self.runtime.sensor_hub.set_mock_tactile_gain(0.06)
                        if hasattr(self.runtime.sensor_hub, "set_mock_joints"):
                            self.runtime.sensor_hub.set_mock_joints(q, gripper=gw_lift)
                        last.gripper_width = gw_lift'''

new_lift = '''                    elif phase == "lift":
                        # Keep closed; nudge joints slightly "up"
                        q = self.runtime.adapter.current_joints().copy()
                        if fault == "joint_limit" and len(q) > 0:
                            # Drive near a caricature joint limit for diagnosis.
                            q[0] = float(q[0]) + 1.2
                        elif len(q) > 1:
                            q[1] = float(q[1]) - 0.04  # shoulder lift proxy
                        self.runtime.adapter.set_joints(q)
                        # Prefer live cmd from this tick (export reads _last_cmd).
                        tgt = cmd if cmd is not None else last
                        if tgt is None:
                            tgt = getattr(self.runtime, "_last_cmd", None)
                        gw_lift = float(getattr(tgt, "gripper_width", 0.01)) if tgt is not None else 0.01
                        force_lift = float(getattr(tgt, "gripper_force", 20.0)) if tgt is not None else 20.0
                        if fault == "grasp_slip":
                            # Blended slip: reopen on lift (hist min shows prior close).
                            gw_lift = 0.08
                        elif fault == "premature_open":
                            # Clear policy/cmd open at lift after a real close.
                            gw_lift = 0.08
                            force_lift = 20.0
                        elif fault == "soft_grasp":
                            # Stay near-closed with slight drift; contact still weak.
                            gw_lift = 0.012
                            force_lift = 1.5
                            if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                self.runtime.sensor_hub.set_mock_tactile_gain(0.06)
                        if hasattr(self.runtime.sensor_hub, "set_mock_joints"):
                            self.runtime.sensor_hub.set_mock_joints(q, gripper=gw_lift)
                        if tgt is not None:
                            tgt.gripper_width = gw_lift
                            tgt.gripper_force = force_lift
                            last = tgt
                            self.runtime._last_cmd = tgt'''

if old_lift not in t:
    raise SystemExit("lift block not found for fix")
t = t.replace(old_lift, new_lift)

# Also ensure grasp soft_grasp updates _last_cmd
old_soft = '''                            # soft_grasp: close geometry OK but contact/force stay weak
                            if fault == "soft_grasp" and last is not None:
                                last.gripper_force = 1.5
                                if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                    self.runtime.sensor_hub.set_mock_tactile_gain(0.06)
                            elif fault == "premature_open" and last is not None:
                                # Real firm close so hist min~0 before open-at-lift
                                last.gripper_force = 20.0
                                if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                    self.runtime.sensor_hub.set_mock_tactile_gain(1.0)'''
new_soft = '''                            # soft_grasp: close geometry OK but contact/force stay weak
                            if fault == "soft_grasp" and last is not None:
                                last.gripper_force = 1.5
                                self.runtime._last_cmd = last
                                if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                    self.runtime.sensor_hub.set_mock_tactile_gain(0.06)
                            elif fault == "premature_open" and last is not None:
                                # Real firm close so hist min~0 before open-at-lift
                                last.gripper_force = 20.0
                                self.runtime._last_cmd = last
                                if hasattr(self.runtime.sensor_hub, "set_mock_tactile_gain"):
                                    self.runtime.sensor_hub.set_mock_tactile_gain(1.0)'''
if old_soft not in t:
    raise SystemExit("soft grasp force block not found")
t = t.replace(old_soft, new_soft)

p.write_text(t, encoding="utf-8")
print("fixed lift/_last_cmd sync")
