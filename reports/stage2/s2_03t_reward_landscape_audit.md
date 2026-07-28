# S2-03T pre-redesign reward landscape audit

Source commit: `8bef297b38051bb9b550e6e97bc7f3ee6370a635`.

Isaac RewardManager applies `raw × weight × dt`; here `dt=0.02 s`.

## State rewards

| State | Actual reward/control-step |
|---|---:|
| `bilateral_gap_60_mm` | `-0.004750424956` |
| `bilateral_gap_30_mm` | `-0.001404258633` |
| `bilateral_gap_25_mm` | `-0.000358300028` |
| `bilateral_gap_10_mm` | `0.006557588823` |
| `bilateral_gap_5_mm` | `0.011730613194` |
| `single_hand_contact_onset` | `-0.013832595194` |
| `bilateral_contact_onset` | `0.051029561672` |
| `bilateral_verify` | `0.051845056065` |
| `safe_hold` | `0.131845056065` |
| `safe_hold_success_terminal` | `2.131845056065` |
| `hard_impact` | `-0.028089704383` |
| `pushing` | `0.047845055265` |
| `timeout_at_25_mm` | `-0.500358300028` |

## Episode returns

| Episode | Return |
|---|---:|
| `hover_25_mm_1000_steps` | `-0.858300027522` |
| `approach_60_to_25_mm_then_hover` | `-1.308792560270` |
| `bilateral_contact_verify_hold` | `16.225484199796` |
| `single_hand_contact_then_timeout` | `-0.296038912361` |
| `hard_impact_immediate` | `-0.028089704383` |
| `contact_then_pushing` | `1.088823648537` |
| `hover_10_mm_1000_steps` | `6.057588823429` |

## Decision

`CURRENT_REWARD_HAS_HOVER_LOCAL_OPTIMUM=YES`

The 10 mm no-contact hover episode has positive return. The absolute-gap term therefore rewards stationary near-contact states repeatedly. This is a development diagnosis; the frozen old scientific result remains `FAIL / NO_QUALIFIED_CONTACT_POLICY`.
