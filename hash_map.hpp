#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

using darray_td = std::vector<upcxx::global_ptr<kmer_pair>>;
using darray_tu = std::vector<upcxx::global_ptr<int>>;

struct HashMap {
    darray_td data;
    darray_tu used;

    size_t my_size;
    size_t og_size;
    size_t total_size;

    size_t size() const noexcept;
    size_t global_size() const noexcept;

    HashMap(size_t size, size_t my_size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer, upcxx::atomic_domain<int>& atom_op_domain);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    int get_target_rank(uint64_t hash);

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot, upcxx::atomic_domain<int>& atom_op_domain);
    bool slot_used(uint64_t slot);
};

int HashMap::get_target_rank(uint64_t hash) {
    return (int)floor(hash/size());    
}

HashMap::HashMap(size_t size, size_t my_size) {
    total_size = size;
    og_size = my_size;
    used.resize(upcxx::rank_n(),0);
    data.resize(upcxx::rank_n(),0);
    int i = 0;
    while (i < upcxx::rank_n()){
        if (i == upcxx::rank_me()) {
            used[i] = upcxx::new_array<int>(my_size);
            data[i] = upcxx::new_array<kmer_pair>(my_size);
        }
        used[i] = upcxx::broadcast(used[i], i).wait();
        data[i] = upcxx::broadcast(data[i], i).wait();
        i++;
    }
}

bool HashMap::insert(const kmer_pair& kmer, upcxx::atomic_domain<int>& atom_op_domain) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % global_size();
        success = request_slot(slot, atom_op_domain);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < global_size());
  return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % global_size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < global_size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { return upcxx::rget(used[get_target_rank(slot)]+slot%size()).wait() != 0; }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { upcxx::rput(kmer, data[get_target_rank(slot)]+slot%size()).wait(); }

kmer_pair HashMap::read_slot(uint64_t slot) { return upcxx::rget(data[get_target_rank(slot)]+slot%size()).wait(); }

bool HashMap::request_slot(uint64_t slot, upcxx::atomic_domain<int>& atom_op_domain) { return atom_op_domain.fetch_add(used[get_target_rank(slot)]+slot%size(), 1, std::memory_order_relaxed).wait()==0; }

size_t HashMap::size() const noexcept { return og_size; }

size_t HashMap::global_size() const noexcept { return total_size; }