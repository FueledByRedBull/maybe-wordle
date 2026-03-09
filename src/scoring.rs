use anyhow::{Result, bail};

pub const WORD_LENGTH: usize = 5;
pub const PATTERN_SPACE: usize = 243;
pub const ALL_GREEN_PATTERN: u8 = 242;

pub fn encode_feedback(values: [u8; WORD_LENGTH]) -> u8 {
    values.iter().enumerate().fold(0u8, |acc, (index, value)| {
        acc + value * 3u8.pow(index as u32)
    })
}

pub fn decode_feedback(pattern: u8) -> [u8; WORD_LENGTH] {
    let mut value = pattern;
    let mut decoded = [0u8; WORD_LENGTH];
    for slot in &mut decoded {
        *slot = value % 3;
        value /= 3;
    }
    decoded
}

pub fn parse_feedback(raw: &str) -> Result<u8> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.len() != WORD_LENGTH {
        bail!("feedback must be 5 characters");
    }

    let mut values = [0u8; WORD_LENGTH];
    for (index, byte) in normalized.bytes().enumerate() {
        values[index] = match byte {
            b'0' | b'b' => 0,
            b'1' | b'y' => 1,
            b'2' | b'g' => 2,
            _ => bail!("feedback must use digits 0/1/2 or letters b/y/g"),
        };
    }
    Ok(encode_feedback(values))
}

pub fn format_feedback_trits(pattern: u8) -> String {
    decode_feedback(pattern)
        .iter()
        .map(|value| char::from(b'0' + *value))
        .collect()
}

pub fn format_feedback_letters(pattern: u8) -> String {
    decode_feedback(pattern)
        .iter()
        .map(|value| match value {
            0 => 'b',
            1 => 'y',
            2 => 'g',
            _ => unreachable!("feedback trits are base-3"),
        })
        .collect()
}

pub fn score_guess(guess: &str, answer: &str) -> u8 {
    let guess = guess.as_bytes();
    let answer = answer.as_bytes();
    debug_assert_eq!(guess.len(), WORD_LENGTH);
    debug_assert_eq!(answer.len(), WORD_LENGTH);

    let mut feedback = [0u8; WORD_LENGTH];
    let mut counts = [0u8; 26];

    for index in 0..WORD_LENGTH {
        if guess[index] == answer[index] {
            feedback[index] = 2;
        } else {
            counts[(answer[index] - b'a') as usize] += 1;
        }
    }

    for index in 0..WORD_LENGTH {
        if feedback[index] == 2 {
            continue;
        }
        let letter_index = (guess[index] - b'a') as usize;
        if counts[letter_index] > 0 {
            feedback[index] = 1;
            counts[letter_index] -= 1;
        }
    }

    encode_feedback(feedback)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_feedback, encode_feedback, format_feedback_letters, format_feedback_trits,
        parse_feedback, score_guess,
    };

    #[test]
    fn encodes_and_decodes_round_trip() {
        let input = [1, 0, 2, 1, 2];
        let pattern = encode_feedback(input);
        assert_eq!(decode_feedback(pattern), input);
    }

    #[test]
    fn parses_trits_and_letters() {
        let pattern = parse_feedback("10202").expect("valid");
        assert_eq!(format_feedback_trits(pattern), "10202");
        assert_eq!(parse_feedback("ybgbg").expect("valid"), pattern);
        assert_eq!(format_feedback_letters(pattern), "ybgbg");
    }

    #[test]
    fn duplicate_letters_match_wordle_behavior() {
        assert_eq!(
            format_feedback_letters(score_guess("lilly", "alley")),
            "ybgbg"
        );
        assert_eq!(
            format_feedback_letters(score_guess("added", "dread")),
            "yybyg"
        );
        assert_eq!(
            format_feedback_letters(score_guess("llama", "banal")),
            "ybyby"
        );
    }

    #[test]
    fn handles_simple_green_case() {
        assert_eq!(
            format_feedback_trits(score_guess("cigar", "cigar")),
            "22222"
        );
    }
}
