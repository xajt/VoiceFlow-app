use log::debug;

/// Filler words to remove from transcription (Polish + English)
const FILLER_WORDS: &[&str] = &[
    // English fillers
    "um",
    "uh",
    "hmm",
    "hmm,",
    "like,",
    "you know,",
    "you know",
    "I mean,",
    "I mean",
    "basically,",
    "basically",
    "actually,",
    "actually",
    "literally,",
    "literally",
    "sort of",
    "kind of",
    "right,",
    "right?",
    // Polish fillers
    "yyy",
    "yyy,",
    "yyy.",
    "eee",
    "eee,",
    "eee.",
    "emm",
    "emm,",
    "emm.",
    "no ale",
    "no to",
    "no więc",
    "wiadomo,",
    "wiadomo",
    "typa",
    "typa,",
    "kijnie",
    "ole",
    "ole,",
    "jakby",
    "jakby,",
    "no dobra",
    "słuchaj",
    "słuchaj,",
];

/// Remove filler words from text
pub fn remove_fillers(text: &str) -> String {
    let mut result = text.to_string();

    for filler in FILLER_WORDS {
        // Case-insensitive replacement
        // Only remove filler if it's a standalone word (surrounded by word boundaries)
        let patterns = [
            // Filler at start of sentence
            format!("{} ", filler),
            // Filler in middle of sentence
            format!(" {} ", filler),
            format!(" {},", filler),
            format!(" {}.", filler),
            // Filler at end of sentence
            format!(" {}", filler),
        ];

        for pattern in &patterns {
            let lower_result = result.to_lowercase();
            let lower_pattern = pattern.to_lowercase();

            if let Some(pos) = lower_result.find(&lower_pattern) {
                // Verify word boundary
                let before_ok = pos == 0
                    || !result
                        .chars()
                        .nth(pos - 1)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let after_end = pos + pattern.len();
                let after_ok = after_end >= result.len()
                    || !result
                        .chars()
                        .nth(after_end)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if before_ok && after_ok {
                    let replacement = match pattern.chars().next() {
                        Some(' ') => " ",
                        _ => "",
                    };
                    result = result.replace(pattern, replacement);
                    break;
                }
            }
        }
    }

    // Clean up double spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    // Clean up leading/trailing spaces
    result = result.trim().to_string();

    result
}

/// Fix capitalization at the start of sentences
pub fn fix_capitalization(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = true;

    for c in text.chars() {
        if capitalize_next && c.is_alphabetic() {
            result.extend(c.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }

        // Capitalize after sentence-ending punctuation
        if c == '.' || c == '!' || c == '?' {
            capitalize_next = true;
        }

        // Also capitalize after newline
        if c == '\n' {
            capitalize_next = true;
        }
    }

    result
}

/// Strip invisible/zero-width characters from text
fn strip_invisible_chars(text: &str) -> String {
    text.chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{200B}' // Zero-width space
                | '\u{200C}' // Zero-width non-joiner
                | '\u{200D}' // Zero-width joiner
                | '\u{FEFF}' // BOM
                | '\u{00AD}' // Soft hyphen
                | '\u{200E}' // LRM
                | '\u{200F}' // RLM
            )
        })
        .collect()
}

/// Main text processing pipeline — applied BEFORE LLM post-processing
pub fn process_text(text: &str) -> String {
    debug!("Text processing pipeline input: '{}'", text);

    let mut result = text.to_string();

    // Step 1: Strip invisible characters
    result = strip_invisible_chars(&result);

    // Step 2: Remove filler words
    let before_fillers = result.clone();
    result = remove_fillers(&result);
    if result != before_fillers {
        debug!("After filler removal: '{}'", result);
    }

    // Step 3: Fix capitalization
    let before_caps = result.clone();
    result = fix_capitalization(&result);
    if result != before_caps {
        debug!("After capitalization fix: '{}'", result);
    }

    // Step 4: Clean up whitespace
    result = result.trim().to_string();
    // Remove trailing whitespace before punctuation
    while let Some(pos) = result.rfind(" .") {
        result = format!("{}{}{}", &result[..pos], ".", &result[pos + 2..]);
    }
    while let Some(pos) = result.rfind(" ,") {
        result = format!("{}{}{}", &result[..pos], ",", &result[pos + 2..]);
    }

    debug!("Text processing pipeline output: '{}'", result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_fillers_english() {
        assert_eq!(
            remove_fillers("um so I was thinking uh about the project"),
            "so I was thinking about the project"
        );
    }

    #[test]
    fn test_remove_fillers_polish() {
        assert_eq!(
            remove_fillers("yyy więc chciałem powiedzieć eee że to działa"),
            "więc chciałem powiedzieć że to działa"
        );
    }

    #[test]
    fn test_fix_capitalization() {
        assert_eq!(
            fix_capitalization("hello world. this is a test. another sentence."),
            "Hello world. This is a test. Another sentence."
        );
    }

    #[test]
    fn test_fix_capitalization_polish() {
        assert_eq!(
            fix_capitalization("cześć. to jest test. kolejne zdanie."),
            "Cześć. To jest test. Kolejne zdanie."
        );
    }

    #[test]
    fn test_process_text_full() {
        let input = "um więc yyy chciałem powiedzieć że to jest test. kolejne zdanie.";
        let result = process_text(input);
        assert!(result.starts_with("Więc"));
        assert!(!result.contains("um"));
        assert!(!result.contains("yyy"));
    }

    #[test]
    fn test_strip_invisible() {
        let input = "hello\u{200B}world";
        assert_eq!(strip_invisible_chars(input), "helloworld");
    }

    #[test]
    fn test_no_double_spaces() {
        let result = remove_fillers("um so um this um works");
        assert!(!result.contains("  "));
    }
}
