use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::ops::DerefMut;
use std::rc::{Rc, Weak};

const MIN_LEAF_STRING_SIZE: usize = 1024;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Color {
    Red,
    Black,
}

#[derive(Copy, Clone, Debug)]
enum Buffer {
    Original,
    Added,
}

#[derive(Debug)]
struct Piece {
    buffer: Buffer,
    start: usize,
    len: usize,
}

pub(crate) struct Node {
    color: Color,
    piece: Piece,
    left_len: usize,
    left: RefCell<Option<Rc<Node>>>,
    right: RefCell<Option<Rc<Node>>>,
    parent: RefCell<Option<Weak<Node>>>,
}

impl Node {
    fn is_red(&self) -> bool {
        self.color == Color::Red
    }
}

pub(crate) struct NodeBuilder {
    color: Color,
    piece: Piece,
    left_len: usize,
    left: RefCell<Option<Rc<Node>>>,
    right: RefCell<Option<Rc<Node>>>,
    parent: RefCell<Option<Weak<Node>>>,
}

impl NodeBuilder {
    fn new(piece: Piece) -> Self {
        NodeBuilder {
            color: Color::Red,
            piece: piece,
            left_len: 0,
            left: RefCell::new(None),
            right: RefCell::new(None),
            parent: RefCell::new(None),
        }
    }

    fn with_left(mut self, node: Option<Rc<Node>>) -> Self {
        self.left = RefCell::new(node);
        self
    }

    fn with_right(mut self, node: Option<Rc<Node>>) -> Self {
        self.right = RefCell::new(node);
        self
    }

    fn with_parent(mut self, node: Option<Weak<Node>>) -> Self {
        self.parent = RefCell::new(node);
        self
    }

    fn build(self) -> Node {
        Node {
            color: self.color,
            piece: self.piece,
            left_len: self.left_len,
            left: self.left,
            right: self.right,
            parent: self.parent,
        }
    }
}

pub(crate) struct PieceTable {
    original: String,
    added: String,
    root: Rc<Node>,
}

fn get_parent(node: Rc<Node>) -> Option<Rc<Node>> {
    if let Some(p) = node.parent.borrow().as_ref() {
        if let Some(p) = p.upgrade().as_ref() {
            Some(Rc::clone(p))
        } else {
            None
        }
    } else {
        None
    }
}

fn get_grandparent(node: Rc<Node>) -> Option<Rc<Node>> {
    if let Some(parent) = get_parent(node) {
        if let Some(gp) = parent.parent.borrow().as_ref() {
            if let Some(gp) = gp.upgrade().as_ref() {
                Some(Rc::clone(gp))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

fn get_sibling(node: Rc<Node>) -> Option<Rc<Node>> {
    let maybe_parent = get_parent(node.clone());
    if let Some(parent) = maybe_parent {
        let has_left = parent.left.borrow().is_some();
        let has_right = parent.right.borrow().is_some();
        if has_left && Rc::ptr_eq(parent.left.borrow().as_ref().unwrap(), &node) {
            parent.right.borrow().clone()
        } else if has_right && Rc::ptr_eq(parent.left.borrow().as_ref().unwrap(), &node) {
            parent.left.borrow().clone()
        } else {
            unreachable!()
        }
    } else {
        None
    }
}

impl PieceTable {
    pub(crate) fn new(original: String) -> Self {
        let len = original.len();
        PieceTable {
            original: original,
            added: String::new(),
            root: Rc::new(
                NodeBuilder::new(Piece {
                    buffer: Buffer::Original,
                    start: 0,
                    len: len,
                })
                .build(),
            ),
        }
    }

    fn rotate_right(&mut self, node: Rc<Node>) {
        let new_node = Rc::clone(&node.left.borrow().as_ref().unwrap());

        *node.left.borrow_mut() = new_node.right.borrow_mut().take();
        *new_node.right.borrow_mut() = Some(Rc::clone(&node));
        *node.parent.borrow_mut() = Some(Rc::downgrade(&new_node));

        if let Some(ref mut left) = *node.left.borrow_mut() {
            *left.parent.borrow_mut() = Some(Rc::downgrade(&node));
        }

        if let Some(parent) = get_parent(node.clone()) {
            let parent_has_left = parent.left.borrow().is_some();
            let parent_has_right = parent.right.borrow().is_some();

            if parent_has_left && Rc::ptr_eq(parent.left.borrow().as_ref().unwrap(), &node) {
                *parent.left.borrow_mut() = Some(Rc::clone(&node));
            } else if parent_has_right && Rc::ptr_eq(parent.right.borrow().as_ref().unwrap(), &node)
            {
                *parent.right.borrow_mut() = Some(Rc::clone(&node));
            } else {
                unreachable!();
            }

            *new_node.parent.borrow_mut() = Some(Rc::downgrade(&parent));
        } else {
            *new_node.parent.borrow_mut() = None;
            self.root = new_node;
        }
    }

    fn rotate_left(&mut self, node: Rc<Node>) {
        let new_node = Rc::clone(&node.right.borrow().as_ref().unwrap());

        *node.right.borrow_mut() = new_node.left.borrow_mut().take();
        *new_node.left.borrow_mut() = Some(Rc::clone(&node));
        *node.parent.borrow_mut() = Some(Rc::downgrade(&new_node));

        if let Some(ref mut right) = *node.right.borrow_mut() {
            *right.parent.borrow_mut() = Some(Rc::downgrade(&node));
        }

        if let Some(parent) = get_parent(node.clone()) {
            let parent_has_left = parent.left.borrow().is_some();
            let parent_has_right = parent.right.borrow().is_some();

            if parent_has_left && Rc::ptr_eq(parent.left.borrow().as_ref().unwrap(), &node) {
                *parent.left.borrow_mut() = Some(Rc::clone(&node));
            } else if parent_has_right && Rc::ptr_eq(parent.right.borrow().as_ref().unwrap(), &node)
            {
                *parent.right.borrow_mut() = Some(Rc::clone(&node));
            } else {
                unreachable!();
            }

            *new_node.parent.borrow_mut() = Some(Rc::downgrade(&parent));
        } else {
            *new_node.parent.borrow_mut() = None;
            self.root = new_node;
        }
    }

    fn insert_right_of(node: Rc<Node>, piece: Piece) {
        assert!(node.right.borrow().is_none());

        // Check whether the node is the root one.
        let maybe_parent = get_parent(node);

        if maybe_parent.is_none() {
            // Node is the root one, so we just insert right of it as red.
            let new_node = Rc::new(NodeBuilder::new(piece).build());
            // *node.right.borrow_mut() = Some(new_node);
            return;
        }

        // let parent = Rc::clone(&node.parent.borrow().as_ref().unwrap().upgrade().unwrap());

        // if parent.is_red() {
        //     let grandparent =
        //         Rc::clone(&parent.parent.borrow().as_ref().unwrap().upgrade().unwrap());
        //     let aunt = Rc::clone(&grandparent.left.borrow().as_ref().unwrap());
        //     if let Some(aunt) =
        //     if aunt.is_red() {
        //         // Color flip
        //     }
        // }

        // // see whether grandparent exists.

        // if let Some(ref mut parent) = *node.parent.borrow_mut() {
        // } else {
        // }

        // if let Some(ref mut weak_parent) = *node.parent.borrow_mut() {
        //     let parent = weak_parent.upgrade().unwrap();
        //     let weak_grandparent = parent.parent.borrow_mut();
        //     if let Some(grandparent) = weak_grandparent.upgrade() {
        //         // let grandparent = parent.parent.borrow_mut();
        //     } else {
        //     }
        // } else {
        // }
    }
}

// impl Table {
//     pub fn new(text: &str) -> Box<Rope> {
//         let (left_text, right_text) = text.split_at(text.len() / 2);

//         if left_text.len() < MIN_LEAF_STRING_SIZE || right_text.len() < MIN_LEAF_STRING_SIZE {
//             // We should not split the rope.
//             Box::new(Rope {
//                 root: Rc::new(Node::Leaf {
//                     string: text.to_owned(),
//                 }),
//             })
//         } else {
//             let root_rope = Box::new(Rope {
//                 root: Node::Internal {
//                     len: left_text.len(),
//                     left_rope: None,
//                     right_rope: None,
//                 },
//             });

//             enum Position<'a> {
//                 Left(&'a str),
//                 Right(&'a str),
//             }

//             let mut stack = Vec::new();
//             stack.push((root_rope, Position::Left(left_text)));
//             stack.push((root_rope, Position::Right(right_text)));

//             while let Some((mut rope, position)) = stack.pop() {
//                 match rope.deref_mut() {
//                     Rope::Node {
//                         len,
//                         left_rope,
//                         right_rope,
//                     } => {
//                         assert!(left_rope.is_none());
//                         assert!(right_rope.is_none());

//                         // Now, we decide whether we should split the
//                     }
//                     _ => unreachable!(),
//                 }
//             }

//             root_rope
//         }
//     }

//     pub fn nth(&self, index: usize) -> Option<char> {
//         None
//     }
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn table_can_be_created() {
        // let _table = Table::new("my sample text".to_owned());
        // assert_eq!(rope.nth(0), Some('m'));
    }
}
