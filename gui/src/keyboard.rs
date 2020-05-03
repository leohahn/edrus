use std::collections::HashMap;
use std::time::{Duration, Instant};
use winit::{event, event::VirtualKeyCode};

#[derive(Debug, Clone)]
pub enum KeyState {
    InitialDelay { start: Instant, delay: Duration },
    Repeat { count: u64 },
    Released,
}

impl KeyState {
    pub fn is_repeat(&self) -> bool {
        match *self {
            KeyState::Repeat { .. } => true,
            _ => false,
        }
    }

    pub fn was_just_pressed(&self) -> bool {
        match *self {
            KeyState::InitialDelay { start, .. } if start.elapsed().as_millis() == 0 => true,
            _ => false,
        }
    }
}

pub struct Keyboard {
    keys: HashMap<VirtualKeyCode, KeyState>,
    initial_delay: Duration,
}

impl Keyboard {
    pub fn new(initial_delay: Duration) -> Self {
        Keyboard {
            keys: HashMap::new(),
            initial_delay: initial_delay,
        }
    }

    fn transition<F>(&mut self, key: VirtualKeyCode, mut func: F)
    where
        F: FnMut(&KeyState) -> KeyState,
    {
        self.keys
            .entry(key)
            .and_modify(|state| {
                let new_state = func(state);
                *state = new_state;
            })
            .or_insert(KeyState::InitialDelay {
                start: std::time::Instant::now(),
                delay: self.initial_delay,
            });
    }

    pub fn key_state(&self, virtual_keycode: &VirtualKeyCode) -> &KeyState {
        self.keys
            .get(virtual_keycode)
            .expect("this call should never fail")
    }

    pub fn update(&mut self, now: Instant, key: VirtualKeyCode, element_state: event::ElementState) {
        let initial_delay = self.initial_delay;
        self.transition(key, |state| match state {
            KeyState::InitialDelay { start, delay } => {
                if element_state == event::ElementState::Released {
                    KeyState::Released
                } else if now.duration_since(*start) >= *delay {
                    KeyState::Repeat { count: 1 }
                } else {
                    state.clone()
                }
            }
            KeyState::Repeat { count } => {
                if element_state == event::ElementState::Released {
                    KeyState::Released
                } else {
                    KeyState::Repeat { count: count + 1 }
                }
            }
            KeyState::Released => {
                if element_state == event::ElementState::Released {
                    state.clone()
                } else {
                    KeyState::InitialDelay {
                        start: now,
                        delay: initial_delay,
                    }
                }
            }
        });
    }
}
