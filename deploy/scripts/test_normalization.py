def test_normalization():
    examples = ['Abu-abu', 'abu-abu', 'Hitam', 'hitam', 'merah', 'Merah', 'Silver', 'silver']
    print(f"{'Asli':<15} | {'Dinormalisasi':<15}")
    print('-' * 35)
    normalized_set = set()
    for ex in examples:
        norm = ex.strip().title()
        print(f'{ex:<15} | {norm:<15}')
        normalized_set.add(norm)
    print('-' * 35)
    print(f'Kategori unik: {len(normalized_set)}')
    print(f'Kategori: {normalized_set}')
if __name__ == '__main__':
    test_normalization()