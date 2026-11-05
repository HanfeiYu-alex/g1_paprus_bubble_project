# Tactile-Informed Whole-Body Box Transport with the Unitree G1

ECE 398, KIMLAB. Alex Yu (alexy5) and team.

A MuJoCo simulation in which a Unitree G1 humanoid holds a box against its
chest with both arms, walks forward, and reports contact force from six
bubble tactile pads throughout.

This document is written for someone who has not seen the project before.
Section 6 covers what would have to change to run any of it on the physical
robot.

---

## 1. What the project is trying to find out

The robot carries a box. Weight inside the box is not evenly distributed and
can shift while it walks. The question is whether the force pattern across
the bubble sensors carries enough information to notice that shift, and
whether the robot can then change how it is holding the box without dropping
it.

Everything here is the groundwork for that question: getting a genuine
whole-body embrace, instrumenting it, and keeping it while the robot walks.

## 2. The sensors

Six channels, matching the hardware layout on the robot in the lab:

| channel | where | sensor geometry in the model |
|---|---|---|
| `chest` | whole front of the torso | 20 mm thick pad over the measured chest surface |
| `back` | whole back of the torso | same, on the rear surface |
| `left_upper_arm` | shoulder to elbow | sleeve around the limb, limb radius + 4 mm |
| `left_lower_arm` | elbow to wrist | same |
| `right_upper_arm` | | |
| `right_lower_arm` | | |

The pad geometry is not hand-entered. `step9_place_sites.py` reads the
robot's own collision geometry, measures the torso's surface and each limb's
radius, and lays the pads on what it measured. The arm pads are 4 mm thicker
than the limb because the physical bubble is a hollow sleeve with a 3-5 mm
wall. The chest pad is 20 mm thick because the physical chest bubble is a
single 20 mm sheet.

`back` is expected to read zero: nothing is behind the robot. It is kept as a
negative control, since a reading there would mean the sensor model is wrong.

## 3. The files

Run them in this order. Each writes what the next one needs.

| file | what it does |
|---|---|
| `step14_inspect_homie.py` | Prints what is in the robot model: joint names and order, actuator limits, the torso's collision envelope, arm segment lengths, and the walking policy's input and output shapes. Run this first on any new model; nothing else should be written from guesswork about it. |
| `step9_place_sites.py` | Adds the six bubble pads to the robot description and writes `g1_bubble.xml`. Also switches the model's articulated hands off, because the robot in the lab does not have them. Ends with a self-check that pushes a box at the chest and confirms the contact lands inside the chest pad. |
| `hug_walk.py` | The main program. Searches for a hugging posture, verifies it under the real controller, then runs: stand, open the arms, embrace, close the forearms, and walk forward, logging all six channels to CSV. |

```
python step14_inspect_homie.py
python step9_place_sites.py --dir <g1_description> --source g1.xml --out g1_bubble.xml
python hug_walk.py --log hug_walk.csv
```

## 4. How the hug is found

The robot is not told what pose to adopt. It is searched for, in two stages.

**Stage 1, geometric.** Joint angles are set directly, one collision pass is
run, and every contact point between the box and the robot is tested against
every pad's sensor zone. About half a millisecond per candidate, so the whole
space -- box height, how hard the box is pressed to the chest, shoulder
pitch, roll and yaw, and elbow -- can be swept exhaustively, around 12,000
combinations.

A candidate is only kept if the contacts are on the right faces of the box:

```
chest      on the box's BACK face   (the box is against the body)
upper arms on the SIDE faces        (the arms come down either side)
forearms   on the FRONT face        (the forearms close round the front)
```

This face test is what separates an embrace from an arm resting on the lid.
Counting how many pads light up is not enough; earlier versions of the search
returned poses with both forearms vertical and the hands above the head,
brushing the box on the way past, which satisfied a contact count and looked
nothing like a hug.

**Stage 2, physical.** The leading candidates are driven with the real PD
controller against gravity and the real sensors are read. A pose the
controller cannot actually hold is rejected here. The winner is the pose
whose *weakest* channel carries the most force, which is the search
expressing the requirement directly: every pad should be loaded, not just
some of them.

## 5. Reading the output

During the run each line reports the six channels, the robot's pelvis
position and height, and three numbers about the walking policy:

```
t= 12.3s [walk  ] 5/5 pads  480.74 N  pelvis x +0.63 z 0.72  box z 0.94
                                      cmd 0.25  |act| 3.108  feet 5
```

* `5/5 pads` -- how many of the five load-bearing channels are above
  threshold (the back pad is excluded).
* `cmd` -- the forward speed being asked of the walking policy.
* `|act|` -- the size of the policy's leg command. If this is constant the
  policy is not responding, which is a different failure from a robot that is
  trying to walk and cannot.
* `feet` -- ground contact points. A walking robot alternates; a constant
  value means both feet are planted.

The summary at the end gives, for each pad, the minimum and peak force over
the walking phase. A pad whose minimum is above threshold held contact for
the whole walk. The CSV holds every frame.

## 6. What would have to change to run this on the robot

Short answer: the arm trajectory transfers directly, the grasp does not, and
the sensors need calibration. In more detail.

### Transfers as it is

* **The hug joint angles.** Shoulder pitch, roll, yaw and elbow, in radians,
  mirrored left and right. They are joint targets and mean the same thing on
  the hardware. They are saved to `hug_pose.json` after every run.
* **The phase schedule.** Settle, open, embrace, wrap, grip, walk, with the
  timings used in simulation. The important ordering is that the upper arms
  reach the box's sides before the forearms close round the front.
* **The walking policy.** `deploy.onnx` is already a deployment artefact;
  OpenHomie ships `HomieDeploy/` with the Unitree SDK for exactly this.
* **Masking the arm joints in the policy's observation.** In simulation the
  policy is told the upper body is at its default. Without that it froze: it
  was trained with the arms near zero, a hug is an input it never saw, and it
  answered with a standing solution and stopped stepping. The same masking
  has to be applied in the hardware deployment script or the same thing will
  happen on the robot.

### Does not transfer, and has to be replaced

* **The box is welded to the torso.** In simulation a constraint holds the
  box in the robot's arms; the arms squeeze it but do not carry it. On the
  robot nothing welds anything. The box is held by contact and friction
  alone, and whether this grip is sufficient is exactly what the simulation
  does not answer. This is the largest gap and should be tested with the
  robot supported and the box at a safe height.
* **The arm gains.** `arm-kp 250` is a simulation torque gain. The real G1
  takes motor commands through the Unitree SDK with its own gain units, and
  the numbers do not carry across. They need to be set conservatively and
  raised while watching the measured force.
* **Contact compliance.** The simulated box is given a soft contact to stand
  in for the deformable bubble membrane. On the robot the bubble provides
  this itself and the parameter is meaningless.
* **Sensor units.** The simulation reports newtons. The hardware reports
  differential-pressure counts from the MPS20N0040D transducers through the
  Teensy. The mass-calibration routine in the `tactile/` package maps raw
  counts to force using a known reference mass; until that is run, simulated
  and measured traces are not comparable.

### Prepare before the hardware session

1. The pose from `hug_pose.json`, and a mapping from our joint names to the
   SDK's motor indices.
2. The reference trace `hug_walk.csv`, to compare the measured six-channel
   pattern against the simulated one for the same box and pose.
3. The box measured in the lab, 27.5 x 20 x 15.7 cm. In the model it is
   placed with the 27.5 cm edge across the body and 15.7 cm front to back,
   because the forearm is 0.10 m and the whole arm reaches 0.29 m from the
   shoulder, so a box much deeper than 17 cm cannot be wrapped at all. A
   20 cm depth was tried and nothing in the search could reach around it.
4. A sequence that adds load in stages: arms only with the robot supported,
   then standing with the box, then walking.

## 7. Current status and known limits

Working:

* The embrace passes both the face test and the geometric wrap test.
* Stage 2 finds poses holding all five load-bearing channels simultaneously.
* The chest and both forearms hold force steadily for the whole walk.

Open:

* **The upper arms carry very little.** Stage 2 measures 19-21 N on them
  against 210 N on the chest and 100 N on each forearm. This is geometry, not
  tuning: the shoulders are 20 cm apart and the box is 27.5 cm wide, so the
  upper arms have to splay outward to clear the box's sides, and splaying
  outward is the same direction as coming off them. A narrower box, under
  about 24 cm, would let the upper arms press inward on the sides naturally.
* **Walking with the hug is fragile.** The policy was trained with the arms
  near zero and no payload. Some poses walk several metres; others freeze the
  gait entirely.
* Nothing yet tests the actual research question. There is no centre-of-mass
  offset parameter and no re-grasp, so no data on how the six channels
  respond to a shifting load.
