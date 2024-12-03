
const MAX_MESSAGE_WIDTH: usize = 128 - 6 - 2;

fn message_print(message: &str) {
    let message_length = message.len();
    match message_length {
        0 => println!("****{}****", "*".repeat(MAX_MESSAGE_WIDTH)),
        1..MAX_MESSAGE_WIDTH => {
            let space = MAX_MESSAGE_WIDTH - message_length;
            let left_space = space / 2;
            let right_space = space - left_space;
            println!(
                "*** {}{}{} ***",
                " ".repeat(left_space),
                message,
                " ".repeat(right_space)
            )
        }
        _ => (),
    }
}